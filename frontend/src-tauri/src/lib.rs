use tauri::Manager;

/// Ensure the overlay window stays fully click-through.
///
/// The overlay must *never* intercept mouse events — doing so makes the whole
/// machine unusable, since the transparent window covers the entire screen.
/// The frontend still calls this (with an `ignore` argument it no longer
/// honours) whenever the hover card shows or hides; we always force
/// `set_ignore_cursor_events(true)` regardless, so clicks always fall through
/// to the app underneath.
#[tauri::command]
fn set_click_through(window: tauri::Window, ignore: bool) -> Result<(), String> {
    let _ = ignore;
    window.set_ignore_cursor_events(true).map_err(|e| e.to_string())
}

/// Return the global cursor position in top-left-origin logical screen points.
///
/// The overlay is click-through (`set_ignore_cursor_events(true)`), so the
/// webview never receives `mousemove` events while the cursor is over another
/// app. The frontend polls this command instead.
///
/// `WebviewWindow::cursor_position` reports *physical* pixels relative to the
/// top-left of the primary monitor; we divide by the window's scale factor so
/// the result is in the same logical screen points the backend's Accessibility
/// lookup expects (the `CGEventGetLocation` / AX API convention). Without this
/// conversion the coordinates would be 2x too large on a Retina display.
#[tauri::command]
fn get_cursor_position(app: tauri::AppHandle) -> (f64, f64) {
    if let Some(window) = app.get_webview_window("main") {
        if let (Ok(pos), Ok(scale)) = (window.cursor_position(), window.scale_factor()) {
            let logical = pos.to_logical::<f64>(scale);
            return (logical.x, logical.y);
        }
    }
    (0.0, 0.0)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![set_click_through, get_cursor_position])
        .setup(|app| {
            let window = app.get_webview_window("main").expect("main window not found");

            // Stretch the overlay to cover the whole primary monitor from its
            // top-left corner. The hover card is positioned with `position:
            // fixed` using global cursor coordinates, so the webview viewport
            // has to line up 1:1 with the screen or the card lands in the wrong
            // place (or off-window entirely).
            if let Some(monitor) = window.current_monitor()? {
                let size = monitor.size();
                let scale = monitor.scale_factor();
                let logical_w = size.width as f64 / scale;
                let logical_h = size.height as f64 / scale;
                window.set_size(tauri::LogicalSize::new(logical_w, logical_h))?;
                window.set_position(tauri::LogicalPosition::new(0.0, 0.0))?;
            }

            // Keep the overlay above everything — including other apps' native
            // fullscreen spaces. A plain always-on-top window sits at
            // NSFloatingWindowLevel and stays on its origin Space, so it
            // disappears the moment you switch to a fullscreened app. Two
            // things fix that on macOS: CanJoinAllSpaces (so the window is
            // drawn into every Space, fullscreen ones included) and a window
            // level above the menu bar. NSPopUpMenuWindowLevel clears
            // fullscreen app content and the menu bar while still staying
            // below screen-saver / critical system alerts.
            window.set_always_on_top(true)?;
            #[cfg(target_os = "macos")]
            {
                use objc2_app_kit::{
                    NSPopUpMenuWindowLevel, NSWindow, NSWindowCollectionBehavior,
                };
                // SAFETY: on macOS `ns_window()` hands back a live NSWindow
                // pointer, and `setup` runs on the main thread.
                let ns_window: &NSWindow =
                    unsafe { &*(window.ns_window()?.cast::<NSWindow>()) };
                ns_window.setLevel(NSPopUpMenuWindowLevel);
                ns_window.setCollectionBehavior(
                    ns_window.collectionBehavior()
                        | NSWindowCollectionBehavior::CanJoinAllSpaces
                        | NSWindowCollectionBehavior::Stationary
                        | NSWindowCollectionBehavior::FullScreenAuxiliary,
                );
            }

            // Start fully click-through: the overlay must never block the
            // app underneath it until a hover card is actually showing.
            window.set_ignore_cursor_events(true)?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
