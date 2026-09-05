use tauri::Manager;
use tauri_plugin_global_shortcut::{Code, Modifiers, Shortcut, ShortcutState};

/// Show the hover card window at ``(x, y)`` and make it capture clicks.
///
/// The card lives in its own small, fixed-size window (see `tauri.conf.json`,
/// label `"card"`) — separate from the full-screen `"main"` overlay, which
/// stays click-through *forever* and is never resized or repositioned. Two
/// earlier approaches broke Save clicks for different reasons: toggling
/// `set_ignore_cursor_events` on the full-screen window let a non-click-
/// through state block clicks anywhere on screen, and shrinking/growing that
/// same window to the card's rect on every hover was visually misaligned
/// with the card's rendered content ("show_card_window"/"hide_card_window",
/// now removed). A dedicated pre-sized window has neither problem: showing
/// it just moves it, so the window frame and the card's content — inset by
/// a fixed 20px margin on every side (see HoverCard.tsx), generous enough
/// that a click at the card's visual edge can never land outside the
/// window — line up exactly, and `"main"` never changes state at all.
///
/// `(x, y)` are logical points and are clamped so the card window never ends
/// up positioned off the primary monitor (using `"main"`'s size, which is
/// already stretched to match it — see `setup`).
///
/// The card is shown but deliberately *not* focused — yanking focus off
/// whatever the user is reading every time a card pops up would be worse than
/// the bug it fixes. For Save clicks to still land on the first click while
/// Mirume is a background app, the card window sets `"acceptFirstMouse": true`
/// (see `tauri.conf.json`); without it macOS swallows that first click as a
/// window-activation click and the card auto-hides before a second one.
#[tauri::command]
fn show_card(app: tauri::AppHandle, x: f64, y: f64) -> Result<(), String> {
    let card = app.get_webview_window("card").ok_or("card window not found")?;
    let scale = card.scale_factor().map_err(|e| e.to_string())?;
    let card_size = card
        .outer_size()
        .map_err(|e| e.to_string())?
        .to_logical::<f64>(scale);

    let (clamped_x, clamped_y) = match app.get_webview_window("main") {
        Some(main) => match main.outer_size() {
            Ok(size) => {
                let monitor = size.to_logical::<f64>(scale);
                let max_x = (monitor.width - card_size.width).max(0.0);
                let max_y = (monitor.height - card_size.height).max(0.0);
                (x.clamp(0.0, max_x), y.clamp(0.0, max_y))
            }
            Err(_) => (x, y),
        },
        None => (x, y),
    };

    card.set_position(tauri::LogicalPosition::new(clamped_x, clamped_y))
        .map_err(|e| e.to_string())?;
    card.show().map_err(|e| e.to_string())?;
    card.set_ignore_cursor_events(false).map_err(|e| e.to_string())
}

/// Hide the hover card window and restore its click-through state.
///
/// `set_ignore_cursor_events(true)` is set *before* `hide()` — belt and
/// braces, since a hidden window shouldn't receive events either way, but a
/// window that somehow got shown again should never do so still capturing
/// clicks.
#[tauri::command]
fn hide_card(app: tauri::AppHandle) -> Result<(), String> {
    let card = app.get_webview_window("card").ok_or("card window not found")?;
    card.set_ignore_cursor_events(true).map_err(|e| e.to_string())?;
    card.hide().map_err(|e| e.to_string())
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

/// Show the review window (word list / flashcards / stats), creating it on
/// first use.
///
/// The window is *not* declared in `tauri.conf.json` — it's a normal opaque
/// window nothing like the overlay, and there's no reason to pay for it until
/// the user asks. Both the Cmd+Shift+M shortcut and the card's "Review" button
/// (via the `open_review_window` command) route through here.
fn show_review_window(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("review") {
        let _ = window.show();
        let _ = window.set_focus();
    } else {
        let _ = tauri::WebviewWindowBuilder::new(
            app,
            "review",
            tauri::WebviewUrl::App("review.html".into()),
        )
        .title("Mirume — Review")
        .inner_size(800.0, 600.0)
        .resizable(true)
        .build();
    }
}

/// Open the review window from the hover card's "Review" button.
///
/// Same effect as pressing Cmd+Shift+M when the window is hidden; unlike the
/// shortcut this never toggles it back off, since a button press is always an
/// "I want to see this" and the user can just close the window.
#[tauri::command]
fn open_review_window(app: tauri::AppHandle) -> Result<(), String> {
    show_review_window(&app);
    Ok(())
}

/// Raise `window` above everything, including other apps' native fullscreen
/// spaces, and make it click-through.
///
/// A plain always-on-top window sits at `NSFloatingWindowLevel` and stays on
/// its origin Space, so it disappears the moment you switch to a fullscreened
/// app. Two things fix that on macOS: `CanJoinAllSpaces` (so the window is
/// drawn into every Space, fullscreen ones included) and a window level above
/// the menu bar. `NSPopUpMenuWindowLevel` clears fullscreen app content and
/// the menu bar while still staying below screen-saver / critical system
/// alerts. Applied to both the full-screen overlay and the (much smaller)
/// hover card window, so the card can appear above a fullscreened Chrome
/// window too.
fn configure_overlay_window(window: &tauri::WebviewWindow) -> tauri::Result<()> {
    window.set_always_on_top(true)?;
    #[cfg(target_os = "macos")]
    {
        use objc2_app_kit::{NSPopUpMenuWindowLevel, NSWindow, NSWindowCollectionBehavior};
        // SAFETY: on macOS `ns_window()` hands back a live NSWindow pointer,
        // and `setup` runs on the main thread.
        let ns_window: &NSWindow = unsafe { &*(window.ns_window()?.cast::<NSWindow>()) };
        ns_window.setLevel(NSPopUpMenuWindowLevel);
        ns_window.setCollectionBehavior(
            ns_window.collectionBehavior()
                | NSWindowCollectionBehavior::CanJoinAllSpaces
                | NSWindowCollectionBehavior::Stationary
                | NSWindowCollectionBehavior::FullScreenAuxiliary,
        );
    }
    window.set_ignore_cursor_events(true)?;
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            show_card,
            hide_card,
            get_cursor_position,
            open_review_window
        ])
        .setup(|app| {
            let main = app.get_webview_window("main").expect("main window not found");

            // Stretch the overlay to cover the whole primary monitor from its
            // top-left corner. The hover card's own window is positioned in
            // *this* coordinate space (see show_card), so it has to line up
            // 1:1 with the screen.
            if let Some(monitor) = main.current_monitor()? {
                let size = monitor.size();
                let scale = monitor.scale_factor();
                let logical_w = size.width as f64 / scale;
                let logical_h = size.height as f64 / scale;
                main.set_size(tauri::LogicalSize::new(logical_w, logical_h))?;
                main.set_position(tauri::LogicalPosition::new(0.0, 0.0))?;
            }
            configure_overlay_window(&main)?;

            // The card window is declared in tauri.conf.json (label "card",
            // fixed 380x500 size — 340x460 of card content plus 20px of
            // click-through-safety padding on every side, see HoverCard.tsx
            // — hidden, resizable in case the user wants more room) and
            // created eagerly alongside "main". It just needs the same
            // above-fullscreen treatment and to start click-through,
            // matching "main"'s permanent state. show_card/hide_card only
            // ever move/show or hide it — never resize it, deliberately:
            // resizing on every hover was the previous design's source of
            // the misalignment bug show_card/hide_card replaced.
            if let Some(card) = app.get_webview_window("card") {
                configure_overlay_window(&card)?;
            }

            // Cmd+Shift+M toggles the separate review window (word list /
            // flashcards / stats) — a normal, non-transparent window, unlike
            // the overlay. Created lazily on first use rather than declared
            // in tauri.conf.json, so it doesn't exist (or cost anything)
            // until the user actually asks for it.
            //
            // The handler fires once per key *transition*, so it runs twice
            // per press — once for ShortcutState::Pressed, once for
            // ::Released — which would open then immediately re-hide the
            // window if left unguarded; only the Pressed event toggles it.
            app.handle().plugin(
                tauri_plugin_global_shortcut::Builder::new()
                    .with_shortcut(Shortcut::new(
                        Some(Modifiers::SUPER | Modifiers::SHIFT),
                        Code::KeyM,
                    ))?
                    .with_handler(|app, _shortcut, event| {
                        if event.state() != ShortcutState::Pressed {
                            return;
                        }
                        match app.get_webview_window("review") {
                            Some(window) if window.is_visible().unwrap_or(false) => {
                                let _ = window.hide();
                            }
                            _ => show_review_window(app),
                        }
                    })
                    .build(),
            )?;

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
