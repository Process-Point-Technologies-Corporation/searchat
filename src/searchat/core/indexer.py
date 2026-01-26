import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
import pyarrow as pa
import pyarrow.parquet as pq

from searchat.core.logging_config import get_logger
from searchat.core.progress import ProgressCallback, NullProgressAdapter
from searchat.models import (
    ConversationRecord,
    MessageRecord,
    IndexStats,
    UpdateStats,
    CONVERSATION_SCHEMA,
    METADATA_SCHEMA
)
from searchat.config import PathResolver, Config

logger = get_logger(__name__)


class ConversationIndexer:
    """
    Indexes conversations from multiple AI coding agents.

    Supported agents:
    - Claude Code: ~/.claude/projects/**/*.jsonl
    - Mistral Vibe: ~/.vibe/logs/session/*.json
    """

    def __init__(self, search_dir: Path, config: Config = None):
        self.path_resolver = PathResolver()
        self.claude_dirs = self.path_resolver.resolve_claude_dirs()
        self.vibe_dirs = self.path_resolver.resolve_vibe_dirs()
        self.search_dir = search_dir
        self.data_dir = search_dir / "data"
        self.conversations_dir = self.data_dir / "conversations"
        self.indices_dir = self.data_dir / "indices"

        if config is None:
            config = Config.load()
        self.config = config

        # Embedder is initialized lazily (first indexing operation).
        self._embedder = None

        self.batch_size = config.embedding.batch_size
        self.chunk_size = 1500
        self.chunk_overlap = 200

        self._ensure_directories()

    def _get_embedder(self):
        """Create and cache the embedding model on first use."""
        if self._embedder is not None:
            return self._embedder

        # Check for GPU availability and warn if not using it
        from searchat.gpu_check import check_and_warn_gpu

        check_and_warn_gpu()

        from sentence_transformers import SentenceTransformer

        device = self.config.embedding.get_device()
        logger.info(f"Initializing embedding model on device: {device}")
        self._embedder = SentenceTransformer(self.config.embedding.model, device=device)
        return self._embedder
    
    def _ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(exist_ok=True)
        self.indices_dir.mkdir(exist_ok=True)
    
    def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.chunk_overlap
        
        if len(text) <= chunk_size:
            return [text]
        
        # Split by sentences (simple approach using common punctuation)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If single sentence exceeds chunk size, split it
            if sentence_size > chunk_size:
                # Add current chunk if it exists
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split long sentence by words
                words = sentence.split()
                temp_chunk = []
                temp_size = 0
                
                for word in words:
                    word_size = len(word) + 1  # +1 for space
                    if temp_size + word_size > chunk_size and temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                        temp_chunk = [word]
                        temp_size = word_size
                    else:
                        temp_chunk.append(word)
                        temp_size += word_size
                
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
            
            # If adding sentence exceeds chunk size, start new chunk
            elif current_size + sentence_size + 1 > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep last sentences for overlap
                overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) > 1 else ''
                current_chunk = [overlap_text, sentence] if overlap_text and overlap > 0 else [sentence]
                current_size = len(overlap_text) + sentence_size + 1 if overlap_text else sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size + 1
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _chunk_by_messages(self, messages: List[MessageRecord], title: str) -> List[Dict]:
        chunks_with_metadata = []
        current_chunk = f"{title}\n\n"
        current_messages = []
        start_message_idx = 0
        
        for idx, msg in enumerate(messages):
            msg_text = f"{msg.content}\n\n"
            
            if len(current_chunk) + len(msg_text) > self.chunk_size and current_messages:
                chunks_with_metadata.append({
                    'text': current_chunk,
                    'start_message_index': start_message_idx,
                    'end_message_index': current_messages[-1]
                })
                
                overlap_size = 0
                overlap_messages = []
                for msg_idx in reversed(current_messages):
                    overlap_msg = messages[msg_idx]
                    overlap_text = f"{overlap_msg.role}: {overlap_msg.content}\n\n"
                    if overlap_size + len(overlap_text) <= self.chunk_overlap:
                        overlap_messages.insert(0, msg_idx)
                        overlap_size += len(overlap_text)
                    else:
                        break
                
                if overlap_messages:
                    current_chunk = ""
                    for msg_idx in overlap_messages:
                        overlap_msg = messages[msg_idx]
                        current_chunk += f"{overlap_msg.role}: {overlap_msg.content}\n\n"
                    start_message_idx = overlap_messages[0]
                    current_messages = overlap_messages.copy()
                else:
                    current_chunk = ""
                    current_messages = []
                    start_message_idx = idx
            
            current_chunk += msg_text
            current_messages.append(idx)
        
        if current_chunk.strip():
            chunks_with_metadata.append({
                'text': current_chunk,
                'start_message_index': start_message_idx,
                'end_message_index': current_messages[-1] if current_messages else 0
            })
        
        return chunks_with_metadata
    
    def index_all(
        self,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> IndexStats:
        """
        Build index from scratch by scanning all JSONL source files.

        SAFETY: If an existing index is detected, this requires force=True
        to prevent accidental data loss. The existing index will be completely
        replaced with data from current source files.

        Args:
            force: Must be True to rebuild when existing index is present.
                   This acknowledges that existing indexed data will be lost
                   if source JSONLs are missing or incomplete.
            progress: Optional progress callback for reporting

        Returns:
            IndexStats with indexing results

        Raises:
            RuntimeError: If existing index found and force=False
        """
        if progress is None:
            progress = NullProgressAdapter()
        has_existing_index = self._has_existing_index()

        if has_existing_index and not force:
            raise RuntimeError(
                "Existing index detected. Full rebuild will REPLACE all indexed data.\n\n"
                "If source JSONLs are missing or incomplete, you will lose conversations.\n\n"
                "Options:\n"
                "  1. Use index_append_only() to safely add new conversations\n"
                "  2. Call index_all(force=True) if you have complete source files\n"
                "  3. Delete index manually: rm -rf <search_dir>/data/\n\n"
                f"Index location: {self.data_dir}"
            )

        if has_existing_index:
            logger.warning(
                f"Force rebuilding index. Existing data at {self.data_dir} will be replaced."
            )
            # Clear existing data
            import shutil
            if self.conversations_dir.exists():
                shutil.rmtree(self.conversations_dir)
            if self.indices_dir.exists():
                shutil.rmtree(self.indices_dir)
            self._ensure_directories()

        start_time = time.time()

        # Phase 1: Discovery
        progress.update_phase("Discovering conversation files")

        # Collect all files first for accurate progress tracking
        all_files = []
        file_metadata = []  # Store (file_path, project_id, agent_type)

        for claude_dir in self.claude_dirs:
            if not claude_dir.exists():
                continue
            for project_dir in claude_dir.iterdir():
                if not project_dir.is_dir():
                    continue
                project_id = project_dir.name
                for json_file in project_dir.glob("*.jsonl"):
                    all_files.append(json_file)
                    file_metadata.append((json_file, project_id, 'claude'))

        for vibe_dir in self.vibe_dirs:
            if not vibe_dir.exists():
                continue
            for json_file in vibe_dir.glob("*.json"):
                all_files.append(json_file)
                file_metadata.append((json_file, "vibe-sessions", 'vibe'))

        # Phase 2: Processing conversations
        progress.update_phase("Processing conversations")

        all_records: List[ConversationRecord] = []
        all_chunks_with_meta: List[Dict] = []
        project_records_map: Dict[str, List[ConversationRecord]] = {}

        for idx, (json_file, project_id, agent_type) in enumerate(file_metadata, 1):
            progress.update_file_progress(idx, len(file_metadata), json_file.name)

            try:
                # Process based on agent type
                if agent_type == 'claude':
                    record = self._process_conversation(json_file, project_id, 0)
                else:
                    record = self._process_vibe_session(json_file, 0)

                # Skip conversations with no messages
                if record.message_count == 0:
                    continue

                all_records.append(record)

                if project_id not in project_records_map:
                    project_records_map[project_id] = []
                project_records_map[project_id].append(record)

                # Collect chunks (will batch encode later)
                chunks_with_meta = self._chunk_by_messages(record.messages, record.title)
                for chunk in chunks_with_meta:
                    chunk['_record'] = record  # Store reference for later
                    all_chunks_with_meta.append(chunk)

            except Exception as e:
                if agent_type == 'claude':
                    raise RuntimeError(f"Failed to process {json_file}: {e}") from e
                else:
                    logger.warning(f"Failed to process Vibe session {json_file}: {e}")
                    continue

        # Phase 3: Generate embeddings
        progress.update_phase("Generating embeddings")
        embeddings_array = self._batch_encode_chunks(all_chunks_with_meta, progress)

        # Build metadata for FAISS index
        metadata: List[Dict] = []
        for chunk_idx, (chunk_meta, embedding) in enumerate(zip(all_chunks_with_meta, embeddings_array)):
            record = chunk_meta['_record']
            metadata.append({
                'vector_id': chunk_idx,
                'conversation_id': record.conversation_id,
                'project_id': record.project_id,
                'chunk_index': chunk_idx,
                'chunk_text': chunk_meta['text'],
                'message_start_index': chunk_meta['start_message_index'],
                'message_end_index': chunk_meta['end_message_index'],
                'created_at': record.created_at
            })

        # Phase 4: Building index
        progress.update_phase("Building search index")
        if len(embeddings_array) > 0:
            self._build_faiss_index(embeddings_array, metadata)

        # Phase 5: Saving
        progress.update_phase("Writing to storage")
        for project_id, records in project_records_map.items():
            # Update embedding_id for records
            embedding_offset = 0
            for record in records:
                record.embedding_id = embedding_offset
                chunks_count = len(self._chunk_by_messages(record.messages, record.title))
                embedding_offset += chunks_count
            self._write_parquet_batch(records, project_id)

        self._write_index_metadata(len(all_records), len(embeddings_array))

        # Update final stats
        progress.update_stats(
            conversations=len(all_records),
            chunks=len(all_chunks_with_meta),
            embeddings=len(embeddings_array)
        )
        progress.finish()

        elapsed = time.time() - start_time
        
        parquet_size = sum(
            f.stat().st_size for f in self.conversations_dir.glob("*.parquet")
        ) / (1024 * 1024)
        
        faiss_path = self.indices_dir / "embeddings.faiss"
        faiss_size = faiss_path.stat().st_size / (1024 * 1024) if faiss_path.exists() else 0
        
        return IndexStats(
            total_conversations=len(all_records),
            total_messages=sum(r.message_count for r in all_records),
            index_time_seconds=elapsed,
            parquet_size_mb=parquet_size,
            faiss_size_mb=faiss_size
        )

    def _batch_encode_chunks(
        self,
        chunks_with_meta: List[Dict],
        progress: ProgressCallback | None = None,
    ) -> np.ndarray:
        """
        Encode chunks in batches for better performance.
        Uses configured batch_size to avoid memory issues.

        Args:
            chunks_with_meta: List of chunks with metadata
            progress: Optional progress callback

        Returns:
            Embeddings array
        """
        if progress is None:
            progress = NullProgressAdapter()

        if not chunks_with_meta:
            return np.array([])

        texts = [chunk['text'] for chunk in chunks_with_meta]

        # Encode in batches manually to report progress
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embedder = self._get_embedder()
            batch_embeddings = embedder.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_embeddings.extend(batch_embeddings)

            # Update progress
            progress.update_embedding_progress(
                current=min(i + self.batch_size, len(texts)),
                total=len(texts)
            )

        return np.array(all_embeddings)

    def _process_conversation(self, json_path: Path, project_id: str, embedding_id: int) -> ConversationRecord:
        with open(json_path, 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f]
        
        file_hash = hashlib.sha256(json_path.read_bytes()).hexdigest()
        
        conversation_id = json_path.stem
        title = 'Untitled'
        for entry in lines:
            content = entry.get('message', {}).get('content', '')
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                text = ' '.join(block.get('text', '') for block in content if block.get('type') == 'text').strip()
            else:
                text = ''
            if text:
                title = text[:100]
                break
        
        messages: List[MessageRecord] = []
        full_text_parts: List[str] = []
        
        for idx, entry in enumerate(lines):
            msg_type = entry.get('type')
            if msg_type not in ('user', 'assistant'):
                continue
            
            role = msg_type
            raw_content = entry.get('message', {}).get('content', '')
            if isinstance(raw_content, str):
                content = raw_content
            elif isinstance(raw_content, list):
                content = '\n\n'.join(block.get('text', '') for block in raw_content if block.get('type') == 'text')
            else:
                content = ''
            
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', content, re.DOTALL)
            has_code = len(code_blocks) > 0
            
            timestamp_str = entry.get('timestamp')
            timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
            
            messages.append(MessageRecord(
                sequence=len(messages),
                role=role,
                content=content,
                timestamp=timestamp,
                has_code=has_code,
                code_blocks=code_blocks
            ))
            
            full_text_parts.append(content)
        
        full_text = "\n\n".join(full_text_parts)
        
        created_at = messages[0].timestamp if messages else datetime.now()
        updated_at = messages[-1].timestamp if messages else datetime.now()
        
        return ConversationRecord(
            conversation_id=conversation_id,
            project_id=project_id,
            file_path=str(json_path),
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            message_count=len(messages),
            messages=messages,
            full_text=full_text,
            embedding_id=embedding_id,
            file_hash=file_hash,
            indexed_at=datetime.now()
        )

    def _detect_agent_format(self, file_path: Path) -> str:
        """
        Detect which AI agent format a file belongs to.

        Args:
            file_path: Path to conversation file

        Returns:
            'claude' for Claude Code JSONL, 'vibe' for Mistral Vibe JSON
        """
        # Check by extension first
        if file_path.suffix == '.jsonl':
            return 'claude'
        elif file_path.suffix == '.json':
            # Check if it's in a Vibe directory
            if '.vibe' in str(file_path):
                return 'vibe'
            # Check file structure
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'metadata' in data and 'messages' in data:
                    return 'vibe'
            except (json.JSONDecodeError, KeyError):
                pass
        return 'unknown'

    def _process_vibe_session(self, json_path: Path, embedding_id: int) -> ConversationRecord:
        """
        Process a Mistral Vibe session JSON file.

        Args:
            json_path: Path to Vibe session JSON
            embedding_id: Embedding ID for this conversation

        Returns:
            ConversationRecord with parsed session data
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        file_hash = hashlib.sha256(json_path.read_bytes()).hexdigest()

        metadata = data.get('metadata', {})
        session_id = metadata.get('session_id', json_path.stem)

        # Extract project from working directory
        env = metadata.get('environment', {})
        working_dir = env.get('working_directory', '')
        project_id = Path(working_dir).name if working_dir else 'vibe-session'

        # Parse timestamps
        start_time_str = metadata.get('start_time')
        end_time_str = metadata.get('end_time')
        created_at = datetime.fromisoformat(start_time_str) if start_time_str else datetime.now()
        updated_at = datetime.fromisoformat(end_time_str) if end_time_str else created_at

        # Parse messages
        messages: List[MessageRecord] = []
        full_text_parts: List[str] = []
        title = 'Untitled Vibe Session'

        for msg in data.get('messages', []):
            role = msg.get('role')

            # Skip system messages and tool responses for search
            if role not in ('user', 'assistant'):
                continue

            content = msg.get('content', '')

            # Skip empty content (tool_calls without text)
            if not content:
                continue

            # Extract title from first user message
            if role == 'user' and title == 'Untitled Vibe Session':
                title = content[:100].replace('\n', ' ').strip()

            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', content, re.DOTALL)
            has_code = len(code_blocks) > 0

            messages.append(MessageRecord(
                sequence=len(messages),
                role=role,
                content=content,
                timestamp=created_at,  # Vibe doesn't have per-message timestamps
                has_code=has_code,
                code_blocks=code_blocks
            ))

            full_text_parts.append(content)

        full_text = "\n\n".join(full_text_parts)

        return ConversationRecord(
            conversation_id=session_id,
            project_id=f"vibe-{project_id}",  # Prefix to distinguish from Claude projects
            file_path=str(json_path),
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            message_count=len(messages),
            messages=messages,
            full_text=full_text,
            embedding_id=embedding_id,
            file_hash=file_hash,
            indexed_at=datetime.now()
        )

    def _process_any_conversation(self, file_path: Path, project_id: str, embedding_id: int) -> ConversationRecord:
        """
        Process a conversation file, auto-detecting the format.

        Args:
            file_path: Path to conversation file
            project_id: Project identifier
            embedding_id: Embedding ID

        Returns:
            ConversationRecord
        """
        agent_format = self._detect_agent_format(file_path)

        if agent_format == 'vibe':
            return self._process_vibe_session(file_path, embedding_id)
        elif agent_format == 'claude':
            return self._process_conversation(file_path, project_id, embedding_id)
        else:
            raise ValueError(f"Unknown conversation format: {file_path}")

    def _write_parquet_batch(self, records: List[ConversationRecord], project_id: str) -> None:
        output_path = self.conversations_dir / f"project_{project_id}.parquet"
        
        data = {
            'conversation_id': [r.conversation_id for r in records],
            'project_id': [r.project_id for r in records],
            'file_path': [r.file_path for r in records],
            'title': [r.title for r in records],
            'created_at': [r.created_at for r in records],
            'updated_at': [r.updated_at for r in records],
            'message_count': [r.message_count for r in records],
            'messages': [
                [
                    {
                        'sequence': m.sequence,
                        'role': m.role,
                        'content': m.content,
                        'timestamp': m.timestamp,
                        'has_code': m.has_code,
                        'code_blocks': m.code_blocks
                    }
                    for m in r.messages
                ]
                for r in records
            ],
            'full_text': [r.full_text for r in records],
            'embedding_id': [r.embedding_id for r in records],
            'file_hash': [r.file_hash for r in records],
            'indexed_at': [r.indexed_at for r in records]
        }
        
        table = pa.Table.from_pydict(data, schema=CONVERSATION_SCHEMA)
        pq.write_table(table, output_path)
    
    def _build_faiss_index(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        if n_vectors < 100:
            index = faiss.IndexFlatL2(dimension)
        else:
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, min(100, n_vectors // 10))
            index.train(embeddings)
        
        index.add(embeddings)
        
        faiss.write_index(index, str(self.indices_dir / "embeddings.faiss"))
        
        metadata_table = pa.Table.from_pylist(metadata, schema=METADATA_SCHEMA)
        pq.write_table(metadata_table, self.indices_dir / "embeddings.metadata.parquet")
    
    def _write_index_metadata(self, total_conversations: int, total_chunks: int) -> None:
        metadata = {
            "version": "1.0.0",
            "schema_version": 1,
            "model_name": self.config.embedding.model,
            "last_updated": datetime.now().isoformat(),
            "total_conversations": total_conversations,
            "total_chunks": total_chunks,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
        
        metadata_path = self.indices_dir / "index_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _has_existing_index(self) -> bool:
        """Check if an existing index is present."""
        metadata_path = self.indices_dir / "index_metadata.json"
        faiss_path = self.indices_dir / "embeddings.faiss"
        has_parquets = any(self.conversations_dir.glob("*.parquet"))

        return metadata_path.exists() or faiss_path.exists() or has_parquets

    def _load_existing_metadata(self) -> Dict:
        metadata_path = self.indices_dir / "index_metadata.json"
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def get_indexed_file_paths(self) -> set:
        """
        Get set of all indexed file paths.

        Returns:
            Set of file path strings that are already in the index
        """
        indexed_paths = set()

        for parquet_file in self.conversations_dir.glob("*.parquet"):
            table = pq.read_table(parquet_file, columns=['file_path'])
            df = table.to_pandas()
            indexed_paths.update(df['file_path'].tolist())

        return indexed_paths

    def index_append_only(
        self,
        file_paths: List[str],
        progress: ProgressCallback | None = None,
    ) -> UpdateStats:
        """
        Append-only indexing for new conversation files.

        SAFETY: This method ONLY adds new data. It never modifies or deletes
        existing indexed data. Safe to call even with orphaned parquet data.

        Supports both Claude Code (.jsonl) and Mistral Vibe (.json) files.

        Args:
            file_paths: List of conversation file paths to index
            progress: Optional progress callback

        Returns:
            UpdateStats with indexing results
        """
        if progress is None:
            progress = NullProgressAdapter()

        progress.update_phase("Processing new conversations")
        start_time = time.time()

        existing_metadata = self._load_existing_metadata()
        if existing_metadata is None:
            raise RuntimeError(
                "No existing index found. Cannot append to non-existent index. "
                "Initial index must exist before append-only mode can be used."
            )

        if existing_metadata.get('model_name') != self.config.embedding.model:
            raise ValueError(
                f"Model mismatch: index uses '{existing_metadata.get('model_name')}', "
                f"config specifies '{self.config.embedding.model}'. "
                "Cannot append with different embedding model."
            )

        # Get already indexed paths to skip duplicates
        existing_paths = self.get_indexed_file_paths()

        # Filter to only truly new files
        new_files = [f for f in file_paths if f not in existing_paths]

        if not new_files:
            return UpdateStats(
                new_conversations=0,
                updated_conversations=0,
                skipped_conversations=len(file_paths),
                update_time_seconds=time.time() - start_time
            )

        # Load existing FAISS index
        faiss_path = self.indices_dir / "embeddings.faiss"
        existing_index = faiss.read_index(str(faiss_path))

        # Load existing metadata
        existing_metadata_table = pq.read_table(
            self.indices_dir / "embeddings.metadata.parquet"
        )
        existing_metadata_df = existing_metadata_table.to_pandas()
        next_vector_id = int(existing_metadata_df['vector_id'].max()) + 1 if len(existing_metadata_df) > 0 else 0

        new_embeddings = []
        new_metadata = []
        new_conversation_records: Dict[str, List[ConversationRecord]] = {}
        processed_count = 0

        for idx, file_path in enumerate(new_files, 1):
            json_path = Path(file_path)

            progress.update_file_progress(idx, len(new_files), json_path.name)

            if not json_path.exists():
                logger.warning(f"File not found, skipping: {file_path}")
                continue

            # Detect format and get project_id
            agent_format = self._detect_agent_format(json_path)
            if agent_format == 'vibe':
                project_id = "vibe-sessions"
            else:
                project_id = json_path.parent.name

            try:
                record = self._process_any_conversation(json_path, project_id, next_vector_id)

                # Skip conversations with no messages
                if record.message_count == 0:
                    continue

                if project_id not in new_conversation_records:
                    new_conversation_records[project_id] = []
                new_conversation_records[project_id].append(record)

                chunks_with_meta = self._chunk_by_messages(record.messages, record.title)

                # Batch encode all chunks for this conversation
                chunk_embeddings = self._batch_encode_chunks(chunks_with_meta, progress)

                for chunk_idx, (chunk_meta, embedding) in enumerate(zip(chunks_with_meta, chunk_embeddings)):
                    new_embeddings.append(embedding)

                    new_metadata.append({
                        'vector_id': next_vector_id,
                        'conversation_id': record.conversation_id,
                        'project_id': record.project_id,
                        'chunk_index': chunk_idx,
                        'chunk_text': chunk_meta['text'],
                        'message_start_index': chunk_meta['start_message_index'],
                        'message_end_index': chunk_meta['end_message_index'],
                        'created_at': record.created_at
                    })
                    next_vector_id += 1

                processed_count += 1

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue

        # Append to FAISS index
        if new_embeddings:
            embeddings_array = np.array(new_embeddings).astype(np.float32)
            existing_index.add(embeddings_array)
            faiss.write_index(existing_index, str(faiss_path))

            # Append to metadata parquet
            new_metadata_table = pa.Table.from_pylist(new_metadata, schema=METADATA_SCHEMA)
            combined_metadata = pa.concat_tables([existing_metadata_table, new_metadata_table])
            pq.write_table(combined_metadata, self.indices_dir / "embeddings.metadata.parquet")

        # Append to conversation parquets
        for project_id, records in new_conversation_records.items():
            project_parquet = self.conversations_dir / f"project_{project_id}.parquet"

            new_record_dicts = []
            for r in records:
                new_record_dicts.append({
                    'conversation_id': r.conversation_id,
                    'project_id': r.project_id,
                    'file_path': r.file_path,
                    'title': r.title,
                    'created_at': r.created_at,
                    'updated_at': r.updated_at,
                    'message_count': r.message_count,
                    'messages': [
                        {
                            'sequence': m.sequence,
                            'role': m.role,
                            'content': m.content,
                            'timestamp': m.timestamp,
                            'has_code': m.has_code,
                            'code_blocks': m.code_blocks
                        }
                        for m in r.messages
                    ],
                    'full_text': r.full_text,
                    'embedding_id': r.embedding_id,
                    'file_hash': r.file_hash,
                    'indexed_at': r.indexed_at
                })

            if project_parquet.exists():
                # Append to existing parquet
                existing_table = pq.read_table(project_parquet)
                new_table = pa.Table.from_pylist(new_record_dicts, schema=CONVERSATION_SCHEMA)
                combined_table = pa.concat_tables([existing_table, new_table])
                pq.write_table(combined_table, project_parquet)
            else:
                # Create new parquet for this project
                new_table = pa.Table.from_pylist(new_record_dicts, schema=CONVERSATION_SCHEMA)
                pq.write_table(new_table, project_parquet)

        # Update index metadata
        existing_index_metadata = self._load_existing_metadata()
        total_conversations = existing_index_metadata['total_conversations'] + processed_count
        total_chunks = existing_index_metadata['total_chunks'] + len(new_embeddings)
        self._write_index_metadata(total_conversations, total_chunks)

        progress.update_stats(
            conversations=processed_count,
            chunks=len(new_embeddings),
            embeddings=len(new_embeddings)
        )
        progress.finish()

        elapsed = time.time() - start_time

        return UpdateStats(
            new_conversations=processed_count,
            updated_conversations=0,
            skipped_conversations=len(file_paths) - processed_count,
            update_time_seconds=elapsed
        )
