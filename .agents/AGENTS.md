# Lumina DMX Project Rules

## PatchNode (Patch Panel) Design & Layout Rules

1. **Header Toggle & Width Preservation**: The entire header area (arrow, icon, and name) must act as a single-click toggle hitbox to expand/collapse the node. No double-click triggers or sub-component overrides. When collapsed, only the height of the PatchNode should shrink to **40** (header height) while its width remains completely unchanged, preserving its full horizontal length.
2. **Compact Strip Height**: The minimum height of the universe strip (`MIN_STRIP_H`) must be kept compact at **36** (not 124) to prevent excessive vertical stretching when the fixture layout is thin (1-2 rows of bars).
3. **Responsive / Flexible Height**: The `UniversePane` wrappers and inner elements must use `flex-1` and `min-h-0` to enable responsive/adaptive resizing inside the patch node. The universe strip itself must use `minHeight: dynamicStripH` (where `MIN_STRIP_H = 36`) to remain compact, ensuring it can shrink down, but will still grow dynamically if there are multiple overlapping fixture rows (stacks).
4. **Stacked Fixture Dragging**: When fixtures are stacked/linked (belonging to the same sub-array in `stacks`), dragging any of them must move all siblings in the same stack by the same DMX channel delta. Sibling fixtures should only be unlinked and dragged individually once unstacked via the UI.
5. **Fixture Groups**: Only use fixture-derived groups (automatically generated from assigned fixture group numbers). Do not include manual group creation input panels, "+" buttons, or ALT+N hotkeys.
6. **Active Group Highlight & Auto-Scroll**: Clicking a group button must center/scroll the view to the first fixture in that group and pulse its border with a cyan glow using the `@keyframes borderPulse` animation. Clicking it again or choosing a different group must toggle/remove the focus state.
7. **Dark Visual Theme**: Maintain dark backgrounds and dark-gray borders. Use `border-zinc-700` for the node's selected state and a dark outline (e.g., `#52525b` zinc-600) for selected fixtures rather than bright light-blue/cyan colors.
