#include <vif.hpp>

using namespace vif;

int vif_main(int argc, char* argv[]) {
    std::string param_file = argv[1];
    std::string output_file = argv[2];

    uint_t n = 1;
    uint_t tseed = 42;
    read_args(argc-2, argv+2, arg_list(n, name(tseed, "seed")));

    vec1d g_mean;
    vec2d g_cov;
    fits::read_table(argv[1], "m", g_mean, "cov", g_cov);

    uint_t npt = g_mean.size();

    matrix::decompose_cholesky d;;
    if (!d.decompose(matrix::as_matrix(g_cov))) {
        error("could not decompose covariance matrix! try adding small sigma^2 to diagonal");
        return 1;
    }

    fits::write("l.fits", d.l.base);

    auto seed = make_seed(tseed);

    vec2d g(n, npt);
    for (uint_t i : range(n)) {
        g(i,_) = g_mean + d.l*randomn(seed, npt);
    }

    fits::write(output_file, g);

    return 0;
}
