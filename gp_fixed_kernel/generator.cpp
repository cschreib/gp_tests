#include <vif.hpp>

using namespace vif;

int vif_main(int argc, char* argv[]) {
    std::string param_file = argv[1];
    std::string output_file = argv[2];

    uint_t n = 1;
    uint_t tseed = 42;
    double length = 1.0;
    double nugget = 0.0;
    std::string method = "inverse"; // "inverse", "cholesky"
    std::string kernel = "sqexp"; // "exp", "sqexp", "nn", "gibbs"
    read_args(argc-2, argv+2, arg_list(n, name(tseed, "seed"), length, nugget, method, kernel));

    auto fkernel = [&](double x1, double x2) {
        if (kernel == "exp") {
            return exp(-abs(x1 - x2)/length);
        } else if (kernel == "sqexp") {
            return exp(-0.5*sqr(x1 - x2)/sqr(length));
        } else if (kernel == "gibbs") {
            auto length_func = [&](double x) {
                return length/sqrt(2.0)*(
                    1.0 - 0.8*exp(-0.5*sqr((x - 3.0)/0.5))
                );
            };

            double l1 = length_func(x1);
            double l2 = length_func(x2);
            return ((2*l1*l2)/(sqr(l1) + sqr(l2)))*exp(-0.5*sqr(x1 - x2)/(sqr(l1) + sqr(l2)));
        } else if (kernel == "nn") {
            x1 = (x1 - 3)/length;
            x2 = (x2 - 3)/length;
            return asin(
                2.0*x1*x2/
                sqrt((1.0 + 2.0*sqr(x1))*(1.0 + 2.0*sqr(x2)))
            );
        } else {
            vif_check(false, "unknown kernel '", kernel, "'");
        }
    };

    vec1d xo, yo, eo, xt;
    fits::read_table(param_file, "x", xo, "y", yo, "ye", eo, "xt", xt);

    uint_t no = xo.size();
    uint_t nt = xt.size();

    matrix::mat2d koo(no,no);
    matrix::mat2d kto(nt,no);
    matrix::mat2d ktt(nt,nt);
    for (uint_t i : range(no))
    for (uint_t j : range(i, no)) {
        koo(i,j) = koo(j,i) = fkernel(xo[i], xo[j]) + (i == j ? nugget + sqr(eo[i]) : 0.0);
    }
    for (uint_t i : range(nt))
    for (uint_t j : range(i, nt)) {
        ktt(i,j) = ktt(j,i) = fkernel(xt[i], xt[j]) + (i == j ? nugget : 0.0);
    }
    for (uint_t i : range(nt))
    for (uint_t j : range(no)) {
        kto(i,j) = fkernel(xt[i], xo[j]);
    }

    vec1d m;
    matrix::mat2d ikoo;
    matrix::mat2d cov;
    double dkoo = 0.0;

    double t = now();
    if (method == "inverse") {
        if (!matrix::invert_symmetric(koo, ikoo)) {
            error("could not invert Kernel matrix");
            return 1;
        }

        matrix::inplace_symmetrize(ikoo);

        m = kto*ikoo*yo;
        dkoo = matrix::determinant(koo);
    } else if (method == "cholesky") {
        matrix::decompose_cholesky d;
        if (!d.decompose(koo)) {
            error("could not Cholesky decompose Kernel matrix");
            return 1;
        }

        ikoo = d.invert();

        m = kto*d.solve(yo);
        dkoo = sqr(d.determinant());
    } else {
        error("unknown inversion method '", method, "'");
        return 1;
    }

    t = now() - t;
    print("time for solving: ", t);

    cov = ktt - kto*ikoo*transpose(kto);
    fits::write("cov.fits", cov.base);

    double logp = -0.5*(total(yo*ikoo*yo) + log(dkoo) + yo.size()*log(2.0*dpi));
    print("log evidence: ", logp);

    matrix::mat2d l;
    matrix::decompose_cholesky d;
    if (!d.decompose(cov)) {
        error("could not decompose covariance matrix! try adding small sigma^2 to diagonal");
        return 1;
    }

    fits::write("l.fits", d.l.base);

    auto seed = make_seed(tseed);

    vec2d yt(n, nt);
    for (uint_t i : range(n)) {
        yt(i,_) = m + d.l*randomn(seed, nt);
    }

    fits::write_table(output_file, "yt", yt, "cov", cov.base, "m", m);

    return 0;
}
