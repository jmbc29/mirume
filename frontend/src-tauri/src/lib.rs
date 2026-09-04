use tauri::Manager;

/// Toggle whether the overlay window intercepts mouse clicks.
///
/// The overlay is click-through by default (`ignore: true`) so it never
/// blocks interaction with whatever app is underneath it. The frontend calls
/// this with `ignore: false` while the hover card is visible, so the user
/// can click its save button, then flips it back to `true` once the card
/// hides.
#[tauri::command]
fn set_click_through(window: tauri::Window, ignore: bool) -> Result<(), String> {
    window.set_ignore_cursor_events(ignore).map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![set_click_through])
        .setup(|app| {
            // Start fully click-through: the overlay must never block the
            // app underneath it until a hover card is actually showing.
            let window = app.get_webview_window("main").expect("main window not found");
            window.set_ignore_cursor_events(true)?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
