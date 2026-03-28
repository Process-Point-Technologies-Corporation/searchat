#[cfg(windows)]
fn main() {
    println!("cargo:rustc-link-lib=Rstrtmgr");
}

#[cfg(not(windows))]
fn main() {}
