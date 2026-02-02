"""Zooming utilities for matplotlib plots.

Provides functions for programmatic zooming and interactive zoom capabilities.
"""

import matplotlib.pyplot as plt
import matplotlib.figure as mpl_fig
from matplotlib.dates import date2num
import pandas as pd
from typing import Tuple, List, Optional


def add_zoom_capabilities(
    fig_or_ax,
    zoom_to: Optional[Tuple[float, float]] = None,
):
    """Add zooming capabilities to a matplotlib figure or axes.

    Args:
        fig_or_ax: matplotlib Figure or Axes object
        zoom_to: Optional (xmin, xmax) tuple to zoom to specific x-range.
                 If None, no programmatic zoom is applied.

    Returns:
        The input figure or axes (for chaining).
    """
    if zoom_to is not None:
        # Handle both Figure and Axes inputs
        if isinstance(fig_or_ax, mpl_fig.Figure):
            # It's a Figure
            for ax in fig_or_ax.axes:
                ax.set_xlim(zoom_to)
        else:
            # Assume it's an Axes
            fig_or_ax.set_xlim(zoom_to)

    return fig_or_ax


def zoom_to_date_range(
    ax,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
):
    """Zoom axes to a specific date range.

    Args:
        ax: matplotlib Axes object
        start_date: Start date (pd.Timestamp or datetime-like)
        end_date: End date (pd.Timestamp or datetime-like)

    Returns:
        The input axes (for chaining).
    """
    # Convert dates to matplotlib numeric format
    start_num = date2num(start_date)
    end_num = date2num(end_date)

    ax.set_xlim(start_num, end_num)
    return ax


def link_axes(axes: List[plt.Axes]):
    """Link x-axes of multiple subplots for synchronized zooming.

    Args:
        axes: List of Axes objects to link.

    Returns:
        List of linked axes (same as input).
    """
    if len(axes) < 2:
        return axes

    # Link x-axes: sharex only works during subplot creation, so we manually sync
    # For now, just use sharex which should work if axes are from same figure
    # This is a minimal implementation
    for ax in axes[1:]:
        ax.sharex(axes[0])

    return axes


def zoom_to_preset(
    ax: plt.Axes,
    preset: str,
    reference_date: Optional[float] = None,
):
    """Zoom axes to a preset range.

    Args:
        ax: matplotlib Axes object
        preset: One of 'last_week', 'last_month', 'last_year', 'all'
        reference_date: Reference point for relative presets (e.g., latest date).
                       If None, uses current x-axis maximum.

    Returns:
        The input axes (for chaining).
    """
    # Get current x-axis limits and data range
    xlim = ax.get_xlim()
    xmin, xmax = xlim

    if reference_date is None:
        reference_date = xmax

    preset = preset.lower()
    if preset == 'all':
        # Reset to autoscale
        ax.autoscale(axis='x')
    elif preset == 'last_week':
        ax.set_xlim(reference_date - 7, reference_date)
    elif preset == 'last_month':
        ax.set_xlim(reference_date - 30, reference_date)
    elif preset == 'last_year':
        ax.set_xlim(reference_date - 365, reference_date)
    else:
        raise ValueError(f"Unknown preset: {preset}. "
                         f"Must be one of 'last_week', 'last_month', 'last_year', 'all'.")

    return ax