# Real released-plugin generation demonstration

The `Released plugin live generation demo` workflow uses the actual published
v0.5.0-rc.1 x64 plugin, service ZIP and unchanged research model. Each download
is checked against its published SHA-256, as is official portable Notepad++
8.9.8.1. The service starts from the extracted ZIP without a repository fallback.

It opens a fresh disposable editor, types a Polish factorial request, sends the
real Generate command, and waits for the plugin's proposal/validation review.
The test requires an actual HTTP 200 model response in the service log. It retains
the task, time-stamped UI status/history events, final review and raw window GIF.
Frames come from the real test editor; no model output is substituted and no
success captions are added. The GIF samples the live window about once a second.

Validation is deliberately unconfigured in this demonstration. Apply must remain
disabled throughout. The model response is never executed or applied to the
editor. This demonstrates a working connection, visible model output and safe
review behavior, **not successful factorial generation or functional validation**.
The checkpoint's retained functional score remains 0/12.

Download `nppai-released-live-demo` from the workflow run to inspect the GIF,
report, event trace and service log. A service failure, missing real response,
review timeout, unavailable screen capture or enabled Apply fails the workflow;
those failures are not relabeled as model success.
