import numpy as np
import pygpufit.gpufit as gf
from reconstruction import Particle

def frame_to_particles(frame, initial_sigma=2.0, method = 0, crop_radius = 4):
    def get_background_stdev(flat_roi, fitted_params):
        gauss = np.empty_like(flat_roi)
        k = int(np.sqrt(gauss.shape[0])) // 2
        denom = 2 * fitted_params[3] ** 2
        i = 0
        for x in range(-k, k + 1):
            for y in range(-k, k + 1):
                gauss[i] = np.exp(-((fitted_params[2] - x) ** 2 + (fitted_params[1] - y) ** 2) / denom)
                i += 1
        gauss = gauss * fitted_params[0] / (np.pi * denom) + fitted_params[4]
        background = flat_roi - gauss
        return np.std(background)

    if len(frame.maxima) == 0:
        return list()
    pxd = frame.load()
    width, height = pxd.shape
    print(frame)
    # Filter maxima and convert to floored int.
    xy = list()
    n_particles = 0
    for i in range(frame.maxima.shape[0]):
        if (0 + crop_radius) < frame.maxima[i, 0] < (width - crop_radius - 1) and (0 + crop_radius) < frame.maxima[i, 1] < (height - crop_radius - 1):
            x, y = np.floor(frame.maxima[i, :]).astype(int)
            xy.append([x, y])
            n_particles += 1

    # Prepare data for gpufit

    data = np.empty((n_particles, (crop_radius * 2 + 1)**2), dtype=np.float32)
    params = np.empty((n_particles, 5), dtype=np.float32)
    for i in range(n_particles):
        x, y = xy[i]
        data[i, :] = pxd[x - crop_radius:x + crop_radius + 1, y - crop_radius:y + crop_radius + 1].flatten()
        initial_offset = data[i, :].min()
        initial_intensity = pxd[x, y] - initial_offset
        params[i, :] = [initial_intensity, crop_radius, crop_radius, initial_sigma, initial_offset]

    estimator = gf.EstimatorID.LSE if method == 0 else gf.EstimatorID.MLE
    constraint_type = np.asarray([gf.ConstraintType.LOWER, gf.ConstraintType.FREE, gf.ConstraintType.FREE, gf.ConstraintType.LOWER_UPPER, gf.ConstraintType.LOWER], dtype=np.int32) # intensity, pos, pos, sigma, offset
    constraint_vals = np.zeros((n_particles, 10), dtype=np.float32)
    constraint_vals[:, 0] = 1.0
    constraint_vals[:, 6] = 1.0
    constraint_vals[:, 7] = 10.0
    parameters, states, chi_squares, number_iterations, execution_time = gf.fit_constrained(data, None, gf.ModelID.GAUSS_2D, params, estimator_id=estimator, max_number_iterations=100, constraint_types=constraint_type, constraints=constraint_vals)

    xy = np.asarray(xy)
    parameters[:, 0] *= 2 * np.pi * parameters[:, 3]**2  # scaling from (maximum value of gaussian) to (number of photons)
    parameters[:, 1:3] -= crop_radius


    print('ratio converged         {:6.2f} %'.format(np.sum(states == 0) / n_particles * 100))
    print('ratio max it. exceeded  {:6.2f} %'.format(np.sum(states == 1) / n_particles * 100))
    print('ratio singular hessian  {:6.2f} %'.format(np.sum(states == 2) / n_particles * 100))
    print('ratio neg curvature MLE {:6.2f} %'.format(np.sum(states == 3) / n_particles * 100))
    print('ratio gpu not read      {:6.2f} %'.format(np.sum(states == 4) / n_particles * 100))

    background_stdev = np.empty(n_particles)
    for i in range(n_particles):
        background_stdev[i] = get_background_stdev(data[i, :], parameters[i, :])

    parameters[:, 1] += xy[:, 1]  # offsetting back into image coordinates rather than crop coordinates
    parameters[:, 2] += xy[:, 0]  # offsetting back into image coordinates rather than crop coordinates
    print("Intensity range", np.amin(parameters[:, 0]), np.amax(parameters[:, 0]))
    print("X range", np.amin(parameters[:, 1]), np.amax(parameters[:, 1]))
    print("Y range", np.amin(parameters[:, 2]), np.amax(parameters[:, 2]))
    print("Sigma range", np.amin(parameters[:, 3]), np.amax(parameters[:, 3]))
    print("Offset range", np.amin(parameters[:, 4]), np.amax(parameters[:, 4]))
    particles = list()
    converged = states == 0
    for i in range(n_particles):
        if converged[i]:
            particles.append(Particle(frame=frame.index, x=parameters[i, 1], y=parameters[i, 2], sigma=parameters[i, 3], intensity=parameters[i, 0], offset=parameters[i, 4], bkgstd=background_stdev[i]))

    return particles # TODO: return fit states as well and show overview of results in particlefitnode.














