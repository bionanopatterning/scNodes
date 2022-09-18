import numpy as np
import pygpufit.gpufit as gf
from reconstruction import Particle

def frame_to_particles(frame, initial_sigma=2.0, method = 0, crop_radius = 4, constraints=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]):
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

    ## Set up constraints
    constraint_type, constraint_values = parse_constraints(constraints, n_particles)
    parameters, states, chi_squares, number_iterations, execution_time = gf.fit_constrained(data, None,
                                                                                            gf.ModelID.GAUSS_2D, params,
                                                                                            estimator_id=estimator,
                                                                                            max_number_iterations=100,
                                                                                            constraint_types=constraint_type,
                                                                                            constraints=constraint_values)

    xy = np.asarray(xy)
    parameters[:, 0] *= 2 * np.pi * parameters[:, 3]**2  # scaling from (maximum value of gaussian) to (number of photons)
    parameters[:, 1:3] -= crop_radius
    #
    #
    # print('ratio converged         {:6.2f} %'.format(np.sum(states == 0) / n_particles * 100))
    # print('ratio max it. exceeded  {:6.2f} %'.format(np.sum(states == 1) / n_particles * 100))
    # print('ratio singular hessian  {:6.2f} %'.format(np.sum(states == 2) / n_particles * 100))
    # print('ratio neg curvature MLE {:6.2f} %'.format(np.sum(states == 3) / n_particles * 100))
    # print('ratio gpu not read      {:6.2f} %'.format(np.sum(states == 4) / n_particles * 100))

    background_stdev = np.empty(n_particles)
    for i in range(n_particles):
        background_stdev[i] = get_background_stdev(data[i, :], parameters[i, :])

    parameters[:, 1] += xy[:, 1]  # offsetting back into image coordinates (rather than crop coordinates)
    parameters[:, 2] += xy[:, 0]  # offsetting back into image coordinates
    particles = list()
    converged = states == 0
    for i in range(n_particles):
        if converged[i] and parameters[i, 3] != constraint_values[0, 7]:
            particles.append(Particle(frame=frame.index, x=parameters[i, 1], y=parameters[i, 2], sigma=parameters[i, 3], intensity=parameters[i, 0], offset=parameters[i, 4], bkgstd=background_stdev[i]))


    return particles # TODO: return fit states as well and show overview of results in particlefitnode.


constraint_type_dict = dict()
constraint_type_dict[(True, True)] = gf.ConstraintType.FREE
constraint_type_dict[(False, True)] = gf.ConstraintType.LOWER
constraint_type_dict[(True, False)] = gf.ConstraintType.UPPER
constraint_type_dict[(False, False)] = gf.ConstraintType.LOWER_UPPER

def parse_constraints(constraints, n_particles):
    c_type_intensity = constraint_type_dict[(constraints[0] == -1.0, constraints[1] == -1.0)]
    c_type_sigma = constraint_type_dict[(constraints[6] == -1.0, constraints[7] == -1.0)]
    c_type_offset = constraint_type_dict[(constraints[8] == -1.0, constraints[9] == -1.0)]

    constraint_types = np.asarray([c_type_intensity, gf.ConstraintType.FREE, gf.ConstraintType.FREE, c_type_sigma, c_type_offset], dtype=np.int32)
    constraint_values = np.zeros((n_particles, 10), dtype=np.float32)
    constraint_values[:, :] = constraints
    print(constraint_values[0, :], "constraint vals")
    print(constraint_types, "constraint_Types")
    return constraint_types, constraint_values












