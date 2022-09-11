import numpy as np
import pygpufit.gpufit as gf
from reconstruction import Particle


def frame_to_particles(frame, initial_sigma, method = 0, crop_radius = 4):
    sqrt2pi = np.sqrt(2 * np.pi)
    pxd = frame.load()
    width, height = pxd.shape

    ## Filter maxima and convert to floored int.
    xy = list()
    n_particles = 0
    for i in range(frame.maxima.shape[0]):
        if (0 + crop_radius) < frame.maxima[i, 0] < (width - crop_radius - 1) and (0 + crop_radius) < frame.maxima[i, 1] < (height - crop_radius - 1):
            x, y = np.floor(frame.maxima[i, :]).astype(np.int)
            xy.append([x, y])
            n_particles += 1

    ## Prepare data for gpufit
    data = np.empty((n_particles, (crop_radius * 2 + 1)**2), dtype=np.float32)
    params = np.empty((n_particles, 5), dtype=np.float32)

    for i in range(n_particles):
        x, y = xy[i]
        data[i, :] = pxd[x - crop_radius:x + crop_radius + 1, y - crop_radius:y + crop_radius + 1].flatten()
        params[i, :] = [pxd[x, y] / sqrt2pi, crop_radius + 0.5, crop_radius + 0.5, initial_sigma, data[i, :].min()]

    estimator = gf.EstimatorID.LSE if method == 0 else gf.EstimatorID.MLE

    constraint_types = np.asarray([gf.ConstraintType.FREE, gf.ConstraintType.FREE, gf.ConstraintType.FREE, gf.ConstraintType.UPPER, gf.ConstraintType.FREE])
    constraint_vals = np.zeros((n_particles, 2*5), dtype=np.float32)
    constraint_vals[:, 7] = 10 * initial_sigma
    parameters, states, chi_squares, number_iterations, execution_time = gf.fit_constrained(data, None, gf.ModelID.GAUSS_2D, params, constraints=constraint_vals, constraint_types=constraint_types, estimator_id=estimator, max_number_iterations=50)
    converged = states == 0
    number_converged = np.sum(converged)
    print('ratio converged         {:6.2f} %'.format(number_converged / n_particles * 100))
    print('ratio max it. exceeded  {:6.2f} %'.format(np.sum(states == 1) / n_particles * 100))
    print('ratio singular hessian  {:6.2f} %'.format(np.sum(states == 2) / n_particles * 100))
    print('ratio neg curvature MLE {:6.2f} %'.format(np.sum(states == 3) / n_particles * 100))
    print('ratio gpu not read      {:6.2f} %'.format(np.sum(states == 4) / n_particles * 100))
    xy = np.asarray(xy)
    if len(parameters.shape) == 1:
        return list()
    parameters[:, 1] += xy[:, 0] - crop_radius
    parameters[:, 2] += xy[:, 1] - crop_radius
    parameters[:, 3] *= frame.pixel_size
    parameters = parameters[converged, :]
    particles = list()
    # TODO: figure out how to get uncertainty (nm) for MLE / LSE estimators.

    # TODO: figure out how ThunderStorm determines bkgstd.
    print("Frame index", frame.index)
    for i in range(number_converged):
        particles.append(Particle(frame.index, parameters[i, 2], parameters[i, 1], parameters[i, 3], parameters[i, 0], parameters[i, 4]))
    return particles















