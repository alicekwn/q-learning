"""Price history visualization for Economics Pricing Q-learning demo."""

from __future__ import annotations

import streamlit as st

__all__ = ["render_price_history"]


def render_price_history(
    price_history: list[tuple[int, float, float]],
    skipped_steps: list[tuple[int, int]],
    step_count: int,
    starting_prices_picked: bool = True,
) -> None:
    """Render 3-row table showing price history for Alice and Bob.

    Args:
        price_history: List of (step_num, p1, p2) tuples representing prices chosen at each step
        skipped_steps: List of (start_step, end_step) tuples for fast-forwarded steps
        step_count: Current total step count
        starting_prices_picked: Whether starting prices have been picked (for randomized mode)
    """
    st.subheader("Price History")

    # Filter out step 0 if starting prices haven't been picked yet (randomized mode)
    if not starting_prices_picked:
        price_history = [item for item in price_history if item[0] != 0]

    if not price_history:
        st.info("No price history yet. Pick starting prices to begin training.")
        return

    # Build display data - interleave normal steps and skipped markers
    display_items = []

    # Create sorted list of all step markers (both normal and skipped)
    all_markers = []

    # Add skipped step markers
    for skip_start, skip_end in skipped_steps:
        all_markers.append(
            {
                "type": "skipped",
                "step_range": (skip_start, skip_end),
                "step_num": skip_start,  # Use start for sorting
            }
        )

    # Add normal step markers
    for step_num, p1, p2 in price_history:
        all_markers.append(
            {
                "type": "normal",
                "step_num": step_num,
                "p1": p1,
                "p2": p2,
            }
        )

    # Sort by step number
    all_markers.sort(key=lambda x: x["step_num"])

    # Process and deduplicate (if a step is both in price_history and skipped_steps, prefer skipped)
    processed_skipped_ranges = set()
    for marker in all_markers:
        if marker["type"] == "skipped":
            skip_range = marker["step_range"]
            if skip_range not in processed_skipped_ranges:
                display_items.append(
                    {
                        "type": "skipped",
                        "step_range": skip_range,
                        "step_num": skip_range[0],  # Use start for sorting
                    }
                )
                processed_skipped_ranges.add(skip_range)
        else:
            # Check if this step is in any skipped range
            step_num = marker["step_num"]
            is_skipped = False
            for skip_start, skip_end in skipped_steps:
                if skip_start <= step_num <= skip_end:
                    is_skipped = True
                    break

            # Only add if not skipped
            if not is_skipped:
                display_items.append(
                    {
                        "type": "normal",
                        "step_num": step_num,
                        "p1": marker["p1"],
                        "p2": marker["p2"],
                    }
                )

    if not display_items:
        st.info("No price history to display.")
        return

    # Reverse the order so newest appears on the left (after the player column)
    display_items_reversed = list(reversed(display_items))

    # Build table data: rows are Alice, Bob, Step; columns are steps
    alice_row = []
    bob_row = []
    step_row = []

    # Fixed column width: calculate width to show 10 columns consistently
    num_visible_cols = 10
    fixed_col_width = "90px"  # Fixed width per column

    for item in display_items_reversed:
        if item["type"] == "skipped":
            skip_start, skip_end = item["step_range"]
            alice_row.append("Skipped")
            bob_row.append("Skipped")
            step_row.append(f"Step {skip_start} - {skip_end}")
        else:
            alice_row.append(f"{item['p1']:.1f}")
            bob_row.append(f"{item['p2']:.1f}")
            step_row.append(f"Step {item['step_num']}")

    # Create HTML table with horizontal scroll, no header, fixed column width
    table_id = f"price_history_table_{id(display_items)}"
    html_table = f"""
    <div id="{table_id}_container" style="width: 100%; overflow-x: auto; overflow-y: hidden; border: 1px solid #ddd; border-radius: 5px;">
        <table style="border-collapse: collapse; table-layout: fixed;">
            <tbody>
                <tr>
                    <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">👩🏼‍💼 Alice</td>
                    {' '.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in alice_row])}
                </tr>
                <tr>
                    <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">🧑🏼‍💼 Bob</td>
                    {' '.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in bob_row])}
                </tr>
                <tr>
                    <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">Step</td>
                    {' '.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; font-size: 0.85em; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in step_row])}
                </tr>
            </tbody>
        </table>
    </div>
    """

    st.markdown(html_table, unsafe_allow_html=True)

    if len(display_items) > num_visible_cols:
        st.caption(
            f"💡 Scroll horizontally to see all steps. The newest steps appear on the left."
        )
    elif step_count > len(display_items):
        st.caption(f"Showing all recorded steps. (Total steps: {step_count})")
