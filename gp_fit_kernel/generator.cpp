#include <vif.hpp>
#include <vif/math/optimize.hpp>

using namespace vif;

int vif_main(int argc, char* argv[]) {
    std::string param_file = argv[1];
    std::string output_file = argv[2];

    uint_t n = 1;
    uint_t tseed = 42;
    double length0 = 3.0;
    double var0 = 2.0;
    double nugget = 0.0;
    std::string method = "inverse"; // "inverse", "cholesky"
    std::string kernel = "sqexp"; // "exp", "sqexp", "nn", "gibbs"
    read_args(argc-2, argv+2, arg_list(n, name(tseed, "seed"), length0, var0, nugget, method, kernel));

    vec1d xo, yo, eo, xt;
    fits::read_table(param_file, "x", xo, "y", yo, "ye", eo, "xt", xt);

    uint_t no = xo.size();
    uint_t nt = xt.size();

    matrix::mat2d koo(no,no);
    matrix::mat2d dlkoo(no,no);
    matrix::mat2d dvkoo(no,no);
    matrix::mat2d ikoo;

    double logpe = 0.0;
    for (uint_t i : range(eo)) {
        logpe += 2.0*log(eo[i]);
    }

    auto fkernel = [&](double x1, double x2, double l, double lv) {
        double k;
        if (kernel == "exp") {
            k = exp(-abs(x1 - x2)/l);
        } else if (kernel == "sqexp") {
            k = exp(-0.5*sqr(x1 - x2)/sqr(l));
        } else if (kernel == "gibbs") {
            auto length_func = [&](double x) {
                return l/sqrt(2.0)*(
                    1.0 - 0.8*exp(-0.5*sqr((x - 3.0)/0.5))
                );
            };

            double l1 = length_func(x1);
            double l2 = length_func(x2);
            k = ((2*l1*l2)/(sqr(l1) + sqr(l2)))*exp(-0.5*sqr(x1 - x2)/(sqr(l1) + sqr(l2)));
        } else if (kernel == "nn") {
            x1 = (x1 - 3)/l;
            x2 = (x2 - 3)/l;
            k = asin(
                2.0*x1*x2/
                sqrt((1.0 + 2.0*sqr(x1))*(1.0 + 2.0*sqr(x2)))
            );
        } else {
            vif_check(false, "unknown kernel '", kernel, "'");
        }

        return exp(lv)*k;
    };

    auto dfkernel_l = [&](double x1, double x2, double l, double lv) {
        double dk;
        if (kernel == "exp") {
            dk = exp(-abs(x1 - x2)/l)*abs(x1 - x2)/pow(l,2);
        } else if (kernel == "sqexp") {
            dk = exp(-0.5*sqr(x1 - x2)/sqr(l))*sqr(x1 - x2)/pow(l,3);
        } else if (kernel == "gibbs") {
            vif_check(false, "unimplemented kernel '", kernel, "'");
        } else if (kernel == "nn") {
            vif_check(false, "unimplemented kernel '", kernel, "'");
        } else {
            vif_check(false, "unknown kernel '", kernel, "'");
        }

        return exp(lv)*dk;
    };

    auto dfkernel_v = [&](double x1, double x2, double l, double lv) {
        return fkernel(x1, x2, l, lv);
    };

    auto get_evidence = [&](vec1d p, minimize_function_output opts) {
        vec1d ret(3);
        double l = p[0];
        double lv = p[1];

        for (uint_t i : range(no))
        for (uint_t j : range(i, no)) {
            koo.safe(i,j) = koo.safe(j,i) = fkernel(xo.safe[i], xo.safe[j], l, lv) +
                (i == j ? nugget + sqr(eo[i]) : 0.0);

            if (opts == minimize_function_output::derivatives || opts == minimize_function_output::all) {
                dlkoo.safe(i,j) = dlkoo.safe(j,i) = dfkernel_l(xo.safe[i], xo.safe[j], l, lv);
                dvkoo.safe(i,j) = dvkoo.safe(j,i) = dfkernel_v(xo.safe[i], xo.safe[j], l, lv);
            }
        }

        double ldkoo = 0.0;
        vec1d ikooyo;
        if (method == "inverse") {
            if (!matrix::invert_symmetric(koo, ikoo)) {
                error("could not invert Kernel matrix");
                return ret;
            }

            matrix::inplace_symmetrize(ikoo);
            ikooyo = ikoo*yo;

            if (opts == minimize_function_output::value || opts == minimize_function_output::all) {
                ldkoo = log(matrix::determinant(koo));
            }
        } else if (method == "cholesky") {
            matrix::decompose_cholesky d;
            if (!d.decompose(koo)) {
                error("could not Cholesky decompose Kernel matrix");
                return ret;
            }

            ikoo = d.invert();
            ikooyo = d.solve(yo);

            if (opts == minimize_function_output::value || opts == minimize_function_output::all) {
                ldkoo = 2.0*d.log_determinant();
            }
        } else {
            error("unknown inversion method '", method, "'");
            return ret;
        }

        if (opts == minimize_function_output::value || opts == minimize_function_output::all) {
            double l1 = total(yo*ikooyo);
            double l2 = ldkoo;
            double l3 = yo.size()*log(2.0*dpi);
            ret[0] = 0.5*(l1 + l2 + l3 + logpe);
            vif_check(is_finite(ret[0]), "log evidence is not finite (", ret[0], " = ",
                l1, " + ", l2, " + ", l3, " + ", logpe, ")");
        }

        if (opts == minimize_function_output::derivatives || opts == minimize_function_output::all) {
            double d1 = -0.5*total(ikooyo*dlkoo*ikooyo);
            double d2 = 0.5*total(diagonal(ikoo*dlkoo));
            ret[1] = d1 + d2;
            vif_check(is_finite(ret[1]), "l derivative is not finite (", ret[1], " = ", d1, " + ", d2, ")");

            d1 = -0.5*total(ikooyo*dvkoo*ikooyo);
            d2 = 0.5*total(diagonal(ikoo*dvkoo));
            ret[2] = d1 + d2;
            vif_check(is_finite(ret[1]), "v derivative is not finite (", ret[2], " = ", d1, " + ", d2, ")");
        }

        return ret;
    };

    double t = now();
    minimize_params opts;
    minimize_result r = minimize_bfgs(opts, vec1d{length0, log(var0)}, get_evidence);
    t = now() - t;
    print("time needed: ", t);
    print("success: ", r.success);
    print("iterations: ", r.niter);
    print("values: ", r.params);
    print("loge: ", -r.value);

    double l = r.params[0];
    double lv = r.params[1];

    matrix::mat2d kto(nt,no);
    matrix::mat2d ktt(nt,nt);

    for (uint_t i : range(nt))
    for (uint_t j : range(i, nt)) {
        ktt.safe(i,j) = ktt.safe(j,i) = fkernel(xt.safe[i], xt.safe[j], l, lv) +
            (i == j ? nugget : 0.0);
    }
    for (uint_t i : range(nt))
    for (uint_t j : range(no)) {
        kto.safe(i,j) = fkernel(xt.safe[i], xo.safe[j], l, lv);
    }

    vec1d m = kto*(ikoo*yo);
    matrix::mat2d cov = ktt - kto*ikoo*transpose(kto);

    matrix::decompose_cholesky d;
    if (!d.decompose(cov)) {
        error("could not decompose covariance matrix! try adding small sigma^2 to diagonal");
        return 1;
    }

    auto seed = make_seed(tseed);

    vec2d yt(n, nt);
    for (uint_t i : range(n)) {
        yt(i,_) = m + d.l*randomn(seed, nt);
    }

    fits::write_table(output_file, "yt", yt, "cov", cov.base, "m", m);

    return 0;
}
