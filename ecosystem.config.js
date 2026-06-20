// DEPRECATED — the backend no longer runs under pm2.
//
// It popped a console window (pm2 launched python, which spawned a console-
// attached child; closing it triggered pm2 to restart → the window reopened).
// The backend now runs as the Windows SERVICE "AgeVisionBackend" (WinSW), which
// has NO console window and starts on boot. See PRODUCTION.md and the service/ folder.
//
//   sc query AgeVisionBackend          # status
//   Restart-Service AgeVisionBackend   # restart (admin)
//
// Empty apps[] so an accidental `pm2 start ecosystem.config.js` does nothing.
module.exports = { apps: [] };
