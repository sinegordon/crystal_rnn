import numpy as np


def magnitude_spectrum(traj, dt=0.002):
    """Compute the magnitude spectrum of atom coordinates over time.

    Args:
        traj: Coordinate trajectory with shape (count_frames, count_atoms * 3).
        dt: Time step between frames in picoseconds.

    Returns:
        A tuple of (frequencies, midpoint_index, averaged_magnitude_spectrum).
    """
    m_spectrum = np.zeros(len(traj[:, 0]))
    for k in range(traj.shape[-1]):
        signal = traj[:, k]
        fft_output = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal), d=dt * 10**-12)
        shifted_fft_output = np.fft.fftshift(fft_output)
        np.fft.fftshift(frequencies)
        m_spectrum += np.abs(shifted_fft_output)
    mid = len(frequencies) // 2
    return frequencies, mid, m_spectrum / len(traj[:, 0])


def processing_jl(k, p, v):
    """Compute the longitudinal current for a wave vector.

    Args:
        k: Wave vector.
        p: Positions array with shape (count_atoms, 3).
        v: Velocities array with shape (count_atoms, 3).

    Returns:
        The complex longitudinal current value for the given frame.
    """
    prods = np.inner(k, p[:, 0:3])
    exps = np.exp(-1j * prods)
    qvl = np.inner(k, v[:, 0:3]) / np.linalg.norm(k)
    sprodsl = qvl * exps
    jl = np.sum(sprodsl)
    return jl


def get_vel(coords, dt):
    p = coords.reshape((coords.shape[0], coords.shape[1] // 3, 3))
    v = (p[2:] - p[:-2]) / 2 / dt
    return v


def get_sqw(coords, dt, step, kmas):
    """Compute the normalized S(q, w) map for the given coordinates.

    Args:
        coords: Coordinate array with shape (count_frames, count_atoms * 3) in angstroms.
        dt: Time step between frames in picoseconds.
        step: Sampling stride across frames.
        kmas: Wave-vector array with shape (count_vectors, 3) in angstrom^-1.

    Returns:
        A tuple of meshgrid arrays (xi, yi) and the normalized intensity map.
    """
    p0 = coords.reshape((coords.shape[0], coords.shape[1] // 3, 3))
    v0 = (p0[2:] - p0[:-2]) / 2 / dt
    p = p0[1:-1]
    p = p[::step]
    v = v0[::step]

    jl = []
    for i in range(v.shape[0]):
        l = []
        for k in kmas:
            l.append(processing_jl(k, p[i], v[i]))
        jl.append(l)

    jl = np.array(jl)
    window = 100
    frame_count = np.shape(jl)[0]
    tom_ev = 1000 / 2.41799 / 10**14
    jlf = np.zeros((window, len(kmas)), dtype="complex")
    jlkw = np.zeros((window, len(kmas)))

    for frame_index in range(frame_count - window):
        jlf[:, :] += jl[frame_index : frame_index + window, :] * np.conj(jl[frame_index, :])

    jlf /= frame_count - window

    for k in range(len(kmas)):
        jlkw[:, k] = np.abs(np.fft.fft(jlf[:, k]))

    wmas = np.fft.fftfreq(window, step * dt * 10**-12) * tom_ev
    kabsmas = np.linalg.norm(kmas, axis=1)
    xi, yi = np.meshgrid(kabsmas, wmas[0 : window // 2])
    jlp = jlkw[0 : window // 2, :]
    jlp_norm = jlp.max(axis=0)
    jlp /= jlp_norm
    return xi, yi, jlp
