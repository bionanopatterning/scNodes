import numpy as np
import pygpufit.gpufit as gf


def frame_to_particles(frame, initial_sigma=2.0, method=0, crop_radius=4, constraints=(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1)):
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
        # note on the above formula: fitted_params[0] is peak_intensity for GPUfit, but we've turned it in to integrated_itensity in the main function,
        # before calling get_background_stdev. So the factor 1/ (np.pi * denom) is there to turn it back into peak intensity.
        background = flat_roi - gauss
        return np.std(background)

    if len(frame.maxima) == 0:
        return list()
    pxd = frame.load()
    width, height = pxd.shape

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

    # Set up constraints
    constraint_type, constraint_values = parse_constraints(constraints, n_particles, crop_radius)
    parameters, states, chi_squares, number_iterations, execution_time = gf.fit_constrained(data, None,
                                                                                            gf.ModelID.GAUSS_2D, params,
                                                                                            estimator_id=estimator,
                                                                                            max_number_iterations=100,
                                                                                            constraint_types=constraint_type,
                                                                                            constraints=constraint_values)
    # TODO: 3d fitting.
    xy = np.asarray(xy)
    parameters[:, 0] *= 2 * np.pi * parameters[:, 3]**2  # scaling from (maximum value of gaussian) to (number of photons)
    parameters[:, 1:3] -= crop_radius
    if np.sum(states == 0) == 0:
        return list()

    background_stdev = np.empty(n_particles)
    for i in range(n_particles):
        background_stdev[i] = get_background_stdev(data[i, :], parameters[i, :])

    parameters[:, 1] += xy[:, 1]  # offsetting back into image coordinates (rather than crop coordinates)
    parameters[:, 2] += xy[:, 0]  # offsetting back into image coordinates
    particles = list()
    converged = states == 0

    ## 230207
    frame_particle_data = dict()
    accepted_particles = np.zeros_like(parameters[:, 3])
    for i in range(parameters[:, 3].shape[0]):
        accepted_particles[i] = converged[i] and parameters[i, 3] != constraint_values[0, 7]
    n_particles = int(accepted_particles.sum(0))

    accepted_particles = accepted_particles.nonzero()
    frame_particle_data["frame"] = list(np.ones(n_particles, dtype=np.float32) * frame.framenr)
    frame_particle_data["x [nm]"] = parameters[accepted_particles, 1].squeeze().tolist()
    frame_particle_data["y [nm]"] = parameters[accepted_particles, 2].squeeze().tolist()
    frame_particle_data["sigma [nm]"] = parameters[accepted_particles, 3].squeeze().tolist()
    frame_particle_data["intensity [counts]"] = parameters[accepted_particles, 0].squeeze().tolist()
    frame_particle_data["offset [counts]"] = parameters[accepted_particles, 4].squeeze().tolist()
    frame_particle_data["bkgstd [counts]"] = background_stdev[accepted_particles].squeeze().tolist()

    ## END 230207
    return frame_particle_data

def frame_to_particles_3d(frame, initial_sigma=2.0, method=0, crop_radius=4, constraints=(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)):
    """
    frame_to_particles_3d differs from frame_to_particles in the model that is used:
    gf.ModelID.GAUSS_2D_ELLIPTIC vs. gf.ModelID.GAUSS_2D
    the parameters for GAUSS_2D_ELLIPTIC are: [amplitude, x, y, sigma x, sigma y, offset]
    """
    def get_background_stdev(flat_roi, fitted_params):
        gauss = np.empty_like(flat_roi)
        k = int(np.sqrt(gauss.shape[0])) // 2
        denom_x = 2 * fitted_params[3] ** 2
        denom_y = 2 * fitted_params[4] ** 2
        fit_x = fitted_params[2]
        fit_y = fitted_params[1]
        i = 0
        for x in range(-k, k + 1):
            for y in range(-k, k + 1):
                gauss[i] = np.exp(-((fit_x - x)**2 / denom_x + (fit_y - y)**2 / denom_y))
                np.exp(-((fit_x - x)**2 / denom_x + (fit_y - y)**2 / denom_y))
                i += 1
        gauss = gauss * fitted_params[0] / (2 * np.pi * fitted_params[3] * fitted_params[4]) + fitted_params[5]
        background = flat_roi - gauss
        return np.std(background)

    if len(frame.maxima) == 0:
        return list()
    pxd = frame.load()
    width, height = pxd.shape

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
    params = np.empty((n_particles, 6), dtype=np.float32)
    for i in range(n_particles):
        x, y = xy[i]
        data[i, :] = pxd[x - crop_radius:x + crop_radius + 1, y - crop_radius:y + crop_radius + 1].flatten()
        initial_offset = data[i, :].min()
        initial_intensity = pxd[x, y] - initial_offset
        params[i, :] = [initial_intensity, crop_radius, crop_radius, initial_sigma, initial_sigma, initial_offset]

    estimator = gf.EstimatorID.LSE if method == 0 else gf.EstimatorID.MLE

    # Set up constraints
    constraint_type, constraint_values = parse_constraints_3d(constraints, n_particles, crop_radius)
    parameters, states, chi_squares, number_iterations, execution_time = gf.fit_constrained(data, None,
                                                                                            gf.ModelID.GAUSS_2D_ELLIPTIC, params,
                                                                                            estimator_id=estimator,
                                                                                            max_number_iterations=100,
                                                                                            constraint_types=constraint_type,
                                                                                            constraints=constraint_values)
    xy = np.asarray(xy)
    parameters[:, 0] *= 2 * np.pi * parameters[:, 3] * parameters[:, 4]  # scaling from (maximum value of gaussian) to integral over that function
    parameters[:, 1:3] -= crop_radius
    if np.sum(states == 0) == 0:
        return list()

    background_stdev = np.empty(n_particles)
    for i in range(n_particles):
        background_stdev[i] = get_background_stdev(data[i, :], parameters[i, :])

    parameters[:, 1] += xy[:, 1]  # offsetting back into image coordinates (rather than crop coordinates)
    parameters[:, 2] += xy[:, 0]  # offsetting back into image coordinates
    particles = list()
    converged = states == 0

    ## 230207
    frame_particle_data = dict()
    accepted_particles = np.zeros_like(parameters[:, 3])
    for i in range(parameters[:, 3].shape[0]):
        accepted_particles[i] = converged[i] and parameters[i, 3] != constraint_values[0, 7]
    n_particles = int(accepted_particles.sum(0))

    frame_particle_data["frame"] = list(np.ones(n_particles, dtype=np.float32) * frame.framenr)
    frame_particle_data["x [nm]"] = parameters[accepted_particles, 1].squeeze().tolist()
    frame_particle_data["y [nm]"] = parameters[accepted_particles, 2].squeeze().tolist()
    frame_particle_data["sigma [nm]"] = parameters[accepted_particles, 3].squeeze().tolist()
    frame_particle_data["intensity [counts]"] = parameters[accepted_particles, 0].squeeze().tolist()
    frame_particle_data["offset [counts]"] = parameters[accepted_particles, 5].squeeze().tolist()
    frame_particle_data["bkgstd [counts]"] = background_stdev[accepted_particles].squeeze().tolist()
    frame_particle_data["sigma2 [nm]"] = parameters[accepted_particles, 4].squeeze().tolist()
    ## END 230207

    return frame_particle_data


constraint_type_dict = dict()
constraint_type_dict[(True, True)] = gf.ConstraintType.FREE
constraint_type_dict[(False, True)] = gf.ConstraintType.LOWER
constraint_type_dict[(True, False)] = gf.ConstraintType.UPPER
constraint_type_dict[(False, False)] = gf.ConstraintType.LOWER_UPPER


def parse_constraints(constraints, n_particles, crop_radius):
    # order: intensity, x, y, sigma, offset
    c_type_intensity = constraint_type_dict[(constraints[0] == -1.0, constraints[1] == -1.0)]
    c_type_sigma = constraint_type_dict[(constraints[6] == -1.0, constraints[7] == -1.0)]
    c_type_offset = constraint_type_dict[(constraints[8] == -1.0, constraints[9] == -1.0)]

    constraint_types = np.asarray([c_type_intensity, gf.ConstraintType.LOWER_UPPER, gf.ConstraintType.LOWER_UPPER, c_type_sigma, c_type_offset], dtype=np.int32)
    constraint_values = np.zeros((n_particles, 10), dtype=np.float32)
    constraint_values[:, :] = constraints
    constraint_values[:, 3] = crop_radius*2 + 1
    constraint_values[:, 5] = crop_radius*2 + 1
    return constraint_types, constraint_values

def parse_constraints_3d(constraints, n_particles, crop_radius):
    # order: intensity, x, y, sigma x, sigma y, offset
    # constraints
    c_type_intensity = constraint_type_dict[(constraints[0] == -1.0, constraints[1] == -1.0)]
    c_type_sigma_x = constraint_type_dict[(constraints[6] == -1.0, constraints[7] == -1.0)]
    c_type_sigma_y = constraint_type_dict[(constraints[8] == -1.0, constraints[9] == -1.0)]
    c_type_offset = constraint_type_dict[(constraints[10] == -1.0, constraints[11] == -1.0)]

    constraint_types = np.asarray([c_type_intensity, gf.ConstraintType.LOWER_UPPER, gf.ConstraintType.LOWER_UPPER, c_type_sigma_x, c_type_sigma_y, c_type_offset], dtype=np.int32)
    constraint_values = np.zeros((n_particles, 12), dtype=np.float32)
    constraint_values[:, :] = constraints
    constraint_values[:, 3] = crop_radius*2 + 1  # maximum for x
    constraint_values[:, 5] = crop_radius*2 + 1  # maximum for y
    return constraint_types, constraint_values
