use super::ModKey;
use crate::utils::MergeWith;

/// Delay before the window focus is considered to be locked-in for Window
/// MRU ordering. For now the delay is not configurable.
pub const DEFAULT_MRU_COMMIT_MS: u64 = 750;

#[derive(Debug, PartialEq)]
pub struct RecentWindows {
    pub on: bool,
    pub mod_key: ModKey,
}

impl Default for RecentWindows {
    fn default() -> Self {
        RecentWindows {
            on: true,
            mod_key: ModKey::Alt,
        }
    }
}

#[derive(knuffel::Decode, Debug, Default, PartialEq)]
pub struct RecentWindowsPart {
    #[knuffel(child)]
    pub on: bool,
    #[knuffel(child)]
    pub off: bool,
    #[knuffel(child, unwrap(argument, str))]
    pub mod_key: Option<ModKey>,
}

impl MergeWith<RecentWindowsPart> for RecentWindows {
    fn merge_with(&mut self, part: &RecentWindowsPart) {
        self.on |= part.on;
        if part.off {
            self.on = false;
        }
        merge_clone!((self, part), mod_key);
    }
}
