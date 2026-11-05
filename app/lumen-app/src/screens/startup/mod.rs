//! Startup Screen (MVVM): Blank infinite canvas backdrop + Left Option Rail + Center VectorCraft
//! Project Hub.

pub mod components;
pub mod startup_event;
pub mod startup_screen;
pub mod startup_state;
pub mod startup_vm;

pub use startup_event::StartupEvent;
pub use startup_screen::StartupScreen;
pub use startup_state::StartupUiState;
pub use startup_vm::StartupViewModel;
