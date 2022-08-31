import torch

EPS = 5e-6


def tensors_about_equal(t1, t2, eps=EPS):
    diff_norm = torch.norm(t1 - t2)
    sum_norm = torch.norm(t1 + t2)
    result = diff_norm < eps * sum_norm
    if not result:
        print("first tensor:\n", t1)
        print("second tensor:\n", t2)
        print("norm of diff:\n", diff_norm)
        print("norm of sum:\n", sum_norm)
    return result


def set_and_check_parameters(model, observations, parameters, number, opt_number):
    """
    Given a freshly constructed model, test that the parameters pass basic tests
    right after the default initialize_parameters() as well as after being set
    to specific values by the client.  When this functions returns, the model
    parameters have been set to `parameters`
    """
    # Make sure we can get the full set of parameters and that they are optimizable
    # and loggable.

    def test_parameters():
        parameters = model.get_parameters()
        assert len(parameters) == number

        optimizable_parameters = model.get_optimizable_parameters()
        assert len(optimizable_parameters) == opt_number
        print("optimizable parameters: ")
        for p in optimizable_parameters:
            print(p)
            assert p.requires_grad == True

        model.log_parameters()

    # Attempt to log before parameter values are set (this should work
    # without raising an exception)
    model.log_parameters()

    # Attempt to initialize the parameters to default values, then test them.
    model.initialize_parameters(observations)
    test_parameters()

    # Attempt to initialize the parameters to user-specified value, then test them.
    model.set_parameters(**parameters)
    test_parameters()
