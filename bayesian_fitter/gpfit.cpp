#include <vif.hpp>

using namespace vif;

int vif_main(int argc, char* argv[]) {
    std::string data_file = argv[1];
    std::string model_file = argv[2];

    vec1d x, y, e, xt;
    fits::read_table(data_file, ftable(x, y, e, xt));


    uint_t n = x.size();
    uint_t nt = xt.size();

    vec1d fx, prior;
    vec2d fy;
    matrix::mat2d prior_cov;
    fits::read_table(model_file, ftable(fx, fy, prior));
    fits::read_table(model_file, "prior_cov", prior_cov.base);

    uint_t N = fy.dims[0];

    matrix::mat2d my(N, n), myt(N, nt);
    for (uint_t i : range(N)) {
        my(i,_)  = interpolate(fy(i,_), fx, x)/e;
        myt(i,_) = interpolate(fy(i,_), fx, xt);
    }

    // Inputs: x, y, e
    // Models: my = model evaluated at x
    //         prior = mean prior value of model i
    //         prior_cov = covariance matrix of prior for models i and j

    // Test: xt
    // Models: myt = models evaluated at xt

    // Normalize by uncertainty to the math below is for uniform uncertainties of unity
    y /= e;
    for (uint_t i : range(N)) {
        my(i,_) /= e;
    }

    // Subtract prior mean from observed data so the math below is for prior with zero mean
    y -= prior*my;

    // Using "kernel trick" to speed things up with large number of models
    // --> deal with Nobs x Nobs matrix
    {
        // Build matrix
        // ============

        // With prior:
        // a[i,j] = sum k,l : my[k,i]*my[l,j]*prior_cov[k,l] + (i == j ? 1.0 : 0.0)
        // No prior:
        // a[i,j] = sum k : my[k,i]*my[k,j] + (i == j ? 1.0 : 0.0)
        matrix::mat2d a = transpose(my)*(prior_cov*my);
        diagonal(a) += 1.0;

        if (!inplace_invert_symmetric(a)) {
            error("could not invert K+I matrix");
            return 1;
        }

        inplace_symmetrize(a);

        // Get model parameters
        // ====================

        // Mean

        // With prior:
        // model_param[i] = sum k,l : prior_cov[i,l]*a[l,k]*y[k]
        // No prior:
        // model_param[i] = sum k : a[i,k]*y[k]
        vec1d mp = prior_cov*my*a*y;

        // Sigma
        matrix::mat2d ms = prior_cov - prior_cov*my*a*transpose(my)*prior_cov;

        // Evaluate at test positions
        // ==========================

        vec1d m = transpose(myt)*mp + prior*myt;
        matrix::mat2d s = transpose(myt)*ms*myt;

        // Save
        // ====

        fits::write_table("result.fits", ftable(xt, m));
        fits::update_table("result.fits", "s", s.base);
    }

    // Normal way to solve the problem
    // --> deal with Nmodel x Nmodel matrix
    {

        // Build matrix
        // ============

        matrix::mat2d iprior_cov;
        if (!invert_symmetric(prior_cov, iprior_cov)) {
            error("could not invert prior_cov matrix");
            return 1;
        }

        inplace_symmetrize(iprior_cov);

        // No prior:
        // a[i,j] = sum k : my[i,k]*my[j,k] + (i == j ? 1.0 : 0.0)
        matrix::mat2d a = my*transpose(my) + iprior_cov;

        if (!inplace_invert_symmetric(a)) {
            error("could not invert A matrix");
            return 1;
        }

        inplace_symmetrize(a);

        // Get model parameters
        // ====================

        vec1d mp = a*my*y;
        matrix::mat2d ms = a;

        // Evaluate at test positions
        // ==========================

        vec1d m = transpose(myt)*mp + prior*myt;
        matrix::mat2d s = transpose(myt)*ms*myt;

        // Save
        // ====

        fits::update_table("result.fits", "m2", m, "s2", s.base);
    }

    return 0;
}
