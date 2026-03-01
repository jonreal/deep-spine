"""Tests for the supraspine perception pipeline: IirBatch filter + DelayBuffer.

Verifies:
1. IirBatch identity: cutoff=0 (no filter) passes signal through unchanged
2. IirBatch lowpass: attenuates high-frequency content
3. DelayBuffer: output is exactly N steps behind input
4. Filter + Delay == Delay + Filter (LTI commutativity)
5. Full pipeline: raw → filter → delay matches expected
6. _make_filter: builds IirBatch from spec dict, None when no spec
7. Three-channel pipeline: proprio, visual, action all filter → delay
"""
import numpy as np
from scipy import signal
from deep_spine.spine.filters.iir import IirBatch
from deep_spine.supraspine.delay import DelayBuffer
from deep_spine.supraspine.mlp_sac import _make_filter

np.random.seed(42)


def test_delay_buffer_exact():
    """DelayBuffer output matches input shifted by exactly delay_steps."""
    delay = 5
    n = 6
    buf = DelayBuffer(delay, init_value=np.zeros(n))
    history = []

    for step in range(50):
        x = np.random.randn(n)
        history.append(x.copy())
        buf.push(x)
        out = buf.sample()

        if step < delay:
            # buffer not full yet — returns init_value
            np.testing.assert_array_equal(out, np.zeros(n))
        else:
            # output should be exactly delay steps ago
            np.testing.assert_array_equal(out, history[step - delay])

    print('  delay_buffer_exact: PASSED')


def test_delay_buffer_zero():
    """Delay=0 passes through immediately."""
    buf = DelayBuffer(0, init_value=np.zeros(3))
    for _ in range(10):
        x = np.random.randn(3)
        buf.push(x)
        np.testing.assert_array_equal(buf.sample(), x)

    print('  delay_buffer_zero: PASSED')


def test_iirbatch_lowpass():
    """IirBatch with butter lowpass attenuates a high-frequency signal."""
    n = 6
    cutoff = 0.1
    b, a = signal.butter(1, cutoff, 'low')
    filt = IirBatch(b, a, (n,))

    # Feed a square wave (rich in high frequencies)
    raw_history = []
    filt_history = []
    for step in range(200):
        x = np.ones(n) if (step // 5) % 2 == 0 else -np.ones(n)
        raw_history.append(x.copy())
        y = filt.filter(x)
        filt_history.append(y.copy())

    raw = np.array(raw_history)
    filt_out = np.array(filt_history)

    # Filtered signal should have smaller variance (smoothed)
    assert filt_out.var() < raw.var(), (
        f'Filter did not attenuate: var_filt={filt_out.var():.4f} >= var_raw={raw.var():.4f}'
    )

    # Filtered should NOT be identical to raw
    assert not np.allclose(raw, filt_out), 'Filter output identical to input'

    print('  iirbatch_lowpass: PASSED')


def test_iirbatch_channels_independent():
    """Each channel in IirBatch maintains independent state."""
    n = 4
    b, a = signal.butter(1, 0.2, 'low')
    filt = IirBatch(b, a, (n,))

    # Feed different signals to each channel
    for step in range(50):
        x = np.zeros(n)
        x[0] = 1.0  # only channel 0 gets input
        y = filt.filter(x)

    # Channel 0 should have nonzero output, others ~0
    final = filt.filter(np.zeros(n))
    assert abs(final[0]) > 1e-3, 'Channel 0 should have state'
    for ch in range(1, n):
        assert abs(final[ch]) < 1e-10, f'Channel {ch} should be zero'

    print('  iirbatch_channels_independent: PASSED')


def test_filter_delay_commutativity():
    """filter(x) → delay == delay → filter(x) for LTI systems."""
    n = 6
    delay = 10
    cutoff = 0.15
    b, a = signal.butter(1, cutoff, 'low')

    # Path A: filter first, then delay
    filt_a = IirBatch(b, a, (n,))
    delay_a = DelayBuffer(delay, init_value=np.zeros(n))

    # Path B: delay first, then filter
    filt_b = IirBatch(b, a, (n,))
    delay_b = DelayBuffer(delay, init_value=np.zeros(n))

    N = 200
    inputs = [np.random.randn(n) for _ in range(N)]
    out_a = []
    out_b = []

    for x in inputs:
        # Path A: filter → delay
        filtered = filt_a.filter(x)
        delay_a.push(filtered)
        out_a.append(delay_a.sample().copy())

        # Path B: delay → filter
        delay_b.push(x)
        delayed = delay_b.sample()
        out_b.append(filt_b.filter(delayed).copy())

    out_a = np.array(out_a)
    out_b = np.array(out_b)

    # After transient (delay + filter settling), outputs should match
    settle = delay + 20
    max_diff = np.max(np.abs(out_a[settle:] - out_b[settle:]))
    assert max_diff < 1e-12, f'Commutativity failed: max_diff={max_diff:.2e}'

    print(f'  filter_delay_commutativity: PASSED (max_diff={max_diff:.2e})')


def test_full_pipeline():
    """Simulate the MlpSAC perception pipeline: raw → filter → delay → sample."""
    n = 6
    delay = 8
    cutoff = 0.2
    b, a = signal.butter(1, cutoff, 'low')

    filt = IirBatch(b, a, (n,))
    buf = DelayBuffer(delay, init_value=np.zeros(n))

    N = 100
    inputs = [np.random.randn(n) for _ in range(N)]
    outputs = []

    for x in inputs:
        filtered = filt.filter(x)
        buf.push(filtered)
        outputs.append(buf.sample().copy())

    # Before delay fills, output should be zeros
    for i in range(delay):
        np.testing.assert_array_equal(outputs[i], np.zeros(n))

    # After delay fills, output should be nonzero (filtered signal is nonzero)
    nonzero_found = False
    for i in range(delay + 5, N):
        if np.any(np.abs(outputs[i]) > 1e-6):
            nonzero_found = True
            break
    assert nonzero_found, 'Pipeline produced all zeros after delay filled'

    # Output should NOT be identical to raw input (filter + delay changed it)
    for i in range(delay, N):
        assert not np.array_equal(outputs[i], inputs[i]), (
            f'Step {i}: output equals raw input — filter/delay not working'
        )

    print('  full_pipeline: PASSED')


def test_make_filter_none():
    """_make_filter returns None for falsy specs."""
    assert _make_filter(None, (6,)) is None
    assert _make_filter(0, (6,)) is None
    assert _make_filter({}, (6,)) is None
    print('  make_filter_none: PASSED')


def test_make_filter_builds():
    """_make_filter builds an IirBatch from a valid spec dict."""
    spec = {'order': 1, 'cutoff': 0.1, 'type': 'low'}
    f = _make_filter(spec, (6,))
    assert isinstance(f, IirBatch)
    assert f.shape == (6,)

    # Higher order
    spec2 = {'order': 2, 'cutoff': 0.3, 'type': 'low'}
    f2 = _make_filter(spec2, (4,))
    assert isinstance(f2, IirBatch)

    print('  make_filter_builds: PASSED')


def _run_channel(n, delay, filter_spec, N=100):
    """Helper: simulate one filter → delay channel. Returns inputs and outputs."""
    filt = _make_filter(filter_spec, (n,))
    buf = DelayBuffer(delay, init_value=np.zeros(n))
    inputs = [np.random.randn(n) for _ in range(N)]
    outputs = []
    for x in inputs:
        raw = x.copy()
        if filt is not None:
            raw = filt.filter(raw)
        buf.push(raw)
        outputs.append(buf.sample().copy())
    return inputs, outputs


def test_proprio_channel():
    """Proprio channel: filter smooths, delay shifts."""
    spec = {'order': 1, 'cutoff': 0.15, 'type': 'low'}
    inputs, outputs = _run_channel(n=6, delay=10, filter_spec=spec)

    # Delay phase: zeros
    for i in range(10):
        np.testing.assert_array_equal(outputs[i], np.zeros(6))

    # After settling, output differs from raw (filtered + delayed)
    diffs = [not np.array_equal(outputs[i], inputs[i]) for i in range(20, 100)]
    assert all(diffs), 'Proprio output should differ from raw input'

    print('  proprio_channel: PASSED')


def test_visual_channel():
    """Visual channel with 2-dim signal (qpos, qvel)."""
    spec = {'order': 1, 'cutoff': 0.2, 'type': 'low'}
    inputs, outputs = _run_channel(n=2, delay=20, filter_spec=spec)

    # Delay phase: zeros
    for i in range(20):
        np.testing.assert_array_equal(outputs[i], np.zeros(2))

    # After settling, nonzero
    nonzero = any(np.any(np.abs(outputs[i]) > 1e-6) for i in range(25, 100))
    assert nonzero, 'Visual channel produced all zeros after delay'

    print('  visual_channel: PASSED')


def test_action_channel():
    """Action channel: filter smooths motor commands."""
    spec = {'order': 1, 'cutoff': 0.1, 'type': 'low'}
    inputs, outputs = _run_channel(n=6, delay=5, filter_spec=spec)

    # Collect filtered outputs after delay fills
    out_arr = np.array(outputs[15:])
    in_arr = np.array(inputs[15:])

    # Filtered should be smoother (lower variance)
    assert out_arr.var() < in_arr.var(), (
        f'Action filter did not smooth: var_out={out_arr.var():.4f} >= var_in={in_arr.var():.4f}'
    )

    print('  action_channel: PASSED')


def test_channel_no_filter():
    """Channel with no filter spec passes raw to delay."""
    n = 6
    delay = 5
    inputs, outputs = _run_channel(n=n, delay=delay, filter_spec=None)

    # After delay fills, output should exactly equal input from delay steps ago
    for i in range(delay, 100):
        np.testing.assert_array_equal(outputs[i], inputs[i - delay])

    print('  channel_no_filter: PASSED')


if __name__ == '__main__':
    print('Perception pipeline tests:')
    test_delay_buffer_exact()
    test_delay_buffer_zero()
    test_iirbatch_lowpass()
    test_iirbatch_channels_independent()
    test_filter_delay_commutativity()
    test_full_pipeline()
    print('')
    print('MlpSAC channel tests:')
    test_make_filter_none()
    test_make_filter_builds()
    test_proprio_channel()
    test_visual_channel()
    test_action_channel()
    test_channel_no_filter()
    print('\nAll tests passed.')
