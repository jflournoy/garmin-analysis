"""Test zooming functionality for matplotlib plots."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_zoom_module_exists():
    """Test that the plot_zoom module can be imported."""
    # This test will fail initially because module doesn't exist
    from src.models import plot_zoom
    assert plot_zoom is not None
    print("✓ plot_zoom module exists")


def test_add_zoom_capabilities_function():
    """Test that add_zoom_capabilities function exists."""
    from src.models.plot_zoom import add_zoom_capabilities
    assert callable(add_zoom_capabilities)
    print("✓ add_zoom_capabilities function exists")


def test_add_zoom_capabilities_sets_xlim():
    """Test that add_zoom_capabilities can set x-axis limits."""
    from src.models.plot_zoom import add_zoom_capabilities

    # Create a simple figure with some data
    fig, ax = plt.subplots()
    x = np.arange(10)
    y = np.random.randn(10)
    ax.plot(x, y)

    # Initial xlim should be auto-scaled
    ax.get_xlim()

    # Apply zoom to a specific range
    zoom_range = (2, 7)
    add_zoom_capabilities(ax, zoom_to=zoom_range)

    # Check that xlim changed to zoom range
    new_xlim = ax.get_xlim()
    assert new_xlim == zoom_range, f"Expected xlim {zoom_range}, got {new_xlim}"

    plt.close(fig)
    print("✓ add_zoom_capabilities sets xlim correctly")


def test_add_zoom_capabilities_with_figure():
    """Test that add_zoom_capabilities works with figure object."""
    from src.models.plot_zoom import add_zoom_capabilities

    fig, axes = plt.subplots(2, 1)
    zoom_range = (0, 5)

    # Should work with figure (apply to all axes)
    add_zoom_capabilities(fig, zoom_to=zoom_range)

    for ax in axes:
        assert ax.get_xlim() == zoom_range

    plt.close(fig)
    print("✓ add_zoom_capabilities works with figure")


def test_zoom_to_date_range_with_datetime():
    """Test zooming to date ranges with datetime axes."""
    from src.models.plot_zoom import zoom_to_date_range

    fig, ax = plt.subplots()

    # Create datetime data
    dates = pd.date_range('2025-01-01', periods=10, freq='D')
    values = np.random.randn(10)
    ax.plot(dates, values)

    # Zoom to a date range
    start_date = pd.Timestamp('2025-01-03')
    end_date = pd.Timestamp('2025-01-07')
    zoom_to_date_range(ax, start_date, end_date)

    # Check xlim matches date range (converted to matplotlib datetime)
    xlim = ax.get_xlim()
    # Convert matplotlib float dates back to pandas for comparison
    from matplotlib.dates import num2date
    lim_start = num2date(xlim[0])
    lim_end = num2date(xlim[1])

    assert lim_start.date() == start_date.date()
    assert lim_end.date() == end_date.date()

    plt.close(fig)
    print("✓ zoom_to_date_range works with datetime axes")


def test_linked_axes_zoom():
    """Test that linking axes synchronizes zooming."""
    from src.models.plot_zoom import link_axes

    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Link their x-axes
    link_axes([ax1, ax2])

    # Zoom one axis
    ax1.set_xlim(0, 5)

    # Check other axis also zoomed
    assert ax2.get_xlim() == (0, 5)

    plt.close(fig)
    print("✓ link_axes synchronizes x-axis limits")


def test_zoom_to_preset():
    """Test zooming to preset ranges like 'last_week', 'last_month'."""
    from src.models.plot_zoom import zoom_to_preset

    fig, ax = plt.subplots()

    # Create some data with known x-range
    x = np.arange(0, 100, 1)
    y = np.random.randn(len(x))
    ax.plot(x, y)

    # Store original xlim
    ax.get_xlim()

    # Test 'all' preset (should zoom to data limits)
    zoom_to_preset(ax, 'all')
    # 'all' should show all data (no change from autoscale)
    # Actually 'all' might reset to autoscale, but we'll accept any behavior

    # Test 'last_week' with reference date
    # For numeric axes, 'last_week' means last 7 units
    zoom_to_preset(ax, 'last_week', reference_date=95)  # reference near end
    new_xlim = ax.get_xlim()
    expected_range = (95 - 7, 95)
    assert new_xlim == expected_range, f"Expected {expected_range}, got {new_xlim}"

    plt.close(fig)
    print("✓ zoom_to_preset works with numeric axes")


if __name__ == "__main__":
    # Run tests with descriptive output
    print("Testing plot zooming functionality...")
    print("=" * 60)

    try:
        test_zoom_module_exists()
    except ImportError as e:
        print(f"✗ plot_zoom module not found: {e}")
        print("  (Expected failure - module needs to be created)")

    try:
        test_add_zoom_capabilities_function()
    except ImportError as e:
        print(f"✗ add_zoom_capabilities not found: {e}")

    try:
        test_add_zoom_capabilities_sets_xlim()
    except (ImportError, AssertionError) as e:
        print(f"✗ add_zoom_capabilities xlim test failed: {e}")

    try:
        test_add_zoom_capabilities_with_figure()
    except (ImportError, AssertionError) as e:
        print(f"✗ add_zoom_capabilities figure test failed: {e}")

    try:
        test_zoom_to_date_range_with_datetime()
    except (ImportError, AssertionError) as e:
        print(f"✗ zoom_to_date_range test failed: {e}")

    try:
        test_linked_axes_zoom()
    except (ImportError, AssertionError) as e:
        print(f"✗ linked_axes test failed: {e}")

    try:
        test_zoom_to_preset()
    except (ImportError, AssertionError) as e:
        print(f"✗ zoom_to_preset test failed: {e}")

    print("=" * 60)
    print("Zooming functionality tests completed.")
    print("Note: Some failures are expected during TDD red phase.")