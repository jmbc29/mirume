use tauri::Manager;

/// Logical (point) size of the overlay's full-screen resting state — the
/// whole primary monitor, from (0, 0). Captured once in `setup` and used by
/// `hide_card_window` to restore the window after `show_card_window` has
/// temporarily shrunk it down to the hover card's own bounds.
struct FullScreenSize {
    width: f64,
    height: f64,
}

/// Shrink the overlay window to the hover card's bounding rect and make it
/// capture clicks, so the card's buttons (e.g. Save) receive them instead of
/// passing them through to the app underneath.
///
/// The overlay normally covers the *entire* primary monitor so the card can
/// be positioned anywhere with `position: fixed` global coordinates — a
/// window that size, toggling `set_ignore_cursor_events(false)` on it, would
/// swallow every click anywhere on screen, not just clicks on the card (this
/// file used to force click-through permanently on for exactly that reason).
/// Shrinking the window down to just the card's rect *first* confines
/// non-click-through to the small area the card actually occupies, then
/// enables it.
///
/// `x`/`y`/`width`/`height` are logical points; `x`/`y` are clamped to the
/// monitor bounds so the card (and the area that stops being click-through)
/// never ends up positioned off-screen.
#[tauri::command]
fn show_card_window(
    window: tauri::Window,
    full_screen: tauri::State<'_, FullScreenSize>,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
) -> Result<(), String> {
    let max_x = (full_screen.width - width).max(0.0);
    let max_y = (full_screen.height - height).max(0.0);
    let clamped_x = x.clamp(0.0, max_x);
    let clamped_y = y.clamp(0.0, max_y);
    window
        .set_size(tauri::LogicalSize::new(width, height))
        .map_err(|e| e.to_string())?;
    window
        .set_position(tauri::LogicalPosition::new(clamped_x, clamped_y))
        .map_err(|e| e.to_string())?;
    window.set_ignore_cursor_events(false).map_err(|e| e.to_string())
}

/// Restore the overlay to its full-screen, click-through resting state.
///
/// Re-enables click-through *before* growing the window back to cover the
/// whole monitor, so there is never an instant where a full-screen window is
/// also non-click-through.
#[tauri::command]
fn hide_card_window(
    window: tauri::Window,
    full_screen: tauri::State<'_, FullScreenSize>,
) -> Result<(), String> {
    window.set_ignore_cursor_events(true).map_err(|e| e.to_string())?;
    window
        .set_size(tauri::LogicalSize::new(full_screen.width, full_screen.height))
        .map_err(|e| e.to_string())?;
    window
        .set_position(tauri::LogicalPosition::new(0.0, 0.0))
        .map_err(|e| e.to_string())
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
        .invoke_handler(tauri::generate_handler![
            show_card_window,
            hide_card_window,
            get_cursor_position
        ])
        .setup(|app| {
            let window = app.get_webview_window("main").expect("main window not found");

            // Stretch the overlay to cover the whole primary monitor from its
            // top-left corner. The hover card is positioned with `position:
            // fixed` using global cursor coordinates, so the webview viewport
            // has to line up 1:1 with the screen or the card lands in the wrong
            // place (or off-window entirely). The same dimensions are stashed
            // as managed state so `hide_card_window` can restore them after
            // `show_card_window` shrinks the window down to the card's rect.
            let mut full_screen = FullScreenSize { width: 0.0, height: 0.0 };
            if let Some(monitor) = window.current_monitor()? {
                let size = monitor.size();
                let scale = monitor.scale_factor();
                full_screen.width = size.width as f64 / scale;
                full_screen.height = size.height as f64 / scale;
                window.set_size(tauri::LogicalSize::new(full_screen.width, full_screen.height))?;
                window.set_position(tauri::LogicalPosition::new(0.0, 0.0))?;
            }
            app.manage(full_screen);

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
