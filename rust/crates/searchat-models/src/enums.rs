use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    VerbatimBm25,
    VerbatimSemantic,
    DistillCore,
    DistillCoreFiles,
    DistillCoreRooms,
    DistillAllFacets,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgorithmType {
    Keyword,
    Semantic,
    Hybrid,
    Adaptive,
    CrossLayer,
    Distill,
}
