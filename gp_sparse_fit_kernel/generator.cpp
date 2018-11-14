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
    double std0 = 1.0;
    double nugget = 0.0;
    uint_t nsparse = 10;
    bool test_evidence = false;
    read_args(argc-2, argv+2, arg_list(
        n, name(tseed, "seed"), length0, var0, std0, nugget, nsparse, test_evidence
    ));

    vec1d xt, yt, et, xs;
    fits::read_table(param_file, "x", xt, "y", yt, "ye", et, "xt", xs);

    uint_t nt = xt.size();
    uint_t ns = xs.size();
    uint_t np = nsparse;

    matrix::mat2d kpp(np,np);
    matrix::mat2d dlkpp(np,np);
    matrix::mat2d dvkpp(np,np);
    vec<1,matrix::mat2d> dpkpp(np);
    for (uint_t i : range(np)) {
        dpkpp[i] = matrix::mat2d(np,np);
    }

    matrix::mat2d ktp(nt,np);
    matrix::mat2d dlktp(nt,np);
    matrix::mat2d dvktp(nt,np);
    vec<1,matrix::mat2d> dpktp(np);
    for (uint_t i : range(np)) {
        dpktp[i] = matrix::mat2d(nt,np);
    }

    vec1d ktt(nt);
    vec1d dvktt(nt);

    // Stuff we want to reuse
    matrix::mat2d iqkpp;
    vec1d wb(np);
    vec1d lambda(nt);

    auto fkernel = [&](double x1, double x2, double l, double lv) {
        return exp(lv - 0.5*sqr(x1 - x2)/sqr(l));
    };

    auto dfkernel_l = [&](double x1, double x2, double l, double lv, double k) {
        return k*sqr(x1 - x2)/pow(l,3);
    };

    auto dfkernel_v = [&](double x1, double x2, double l, double lv, double k) {
        return k;
    };

    auto dfkernel_p = [&](double x1, double x2, double l, double lv, double k) {
        return k*(x1 - x2)/sqr(l);
    };

    auto get_evidence = [&](vec1d p, minimize_function_output opts) {
        vec1d ret(1+p.size());
        vec1d xp = p[_-(np-1)];
        double ll = p[np];
        double lv = p[np+1];
        double ls = p[np+2];

        // Evaluate K matrices and derivatives
        double ds = 2*exp(2*ls);

        // Kpp
        for (uint_t i : range(np))
        for (uint_t j : range(i, np)) {
            double k = fkernel(xp.safe[i], xp.safe[j], ll, lv);
            kpp.safe(i,j) = kpp.safe(j,i) = k + (i == j ? nugget + sqr(et[i]) + exp(2*ls) : 0.0);

            if (opts == minimize_function_output::derivatives || opts == minimize_function_output::all) {
                dlkpp.safe(i,j) = dlkpp.safe(j,i) = dfkernel_l(xp.safe[i], xp.safe[j], ll, lv, k);
                dvkpp.safe(i,j) = dvkpp.safe(j,i) = dfkernel_v(xp.safe[i], xp.safe[j], ll, lv, k);

                double dp = dfkernel_p(xp.safe[i], xp.safe[j], ll, lv, k);
                dpkpp.safe[i].safe(i,j) = dpkpp.safe[i].safe(j,i) = -dp;
                dpkpp.safe[j].safe(i,j) = dpkpp.safe[j].safe(j,i) = +dp;
            }
        }

        // Ktt
        for (uint_t i : range(nt)) {
            double k = fkernel(xt.safe[i], xt.safe[i], ll, lv);
            ktt.safe[i] = k + nugget + sqr(et[i]) + exp(2*ls);

            if (opts == minimize_function_output::derivatives || opts == minimize_function_output::all) {
                // dlktt(i,i) = 0

                dvktt.safe[i] = dfkernel_v(xt.safe[i], xt.safe[i], ll, lv, k);

                // dpktt = 0
            }
        }

        // Ktp
        for (uint_t i : range(nt))
        for (uint_t j : range(np)) {
            double k = fkernel(xt.safe[i], xp.safe[j], ll, lv);
            ktp.safe(i,j) = k;

            if (opts == minimize_function_output::derivatives || opts == minimize_function_output::all) {
                dlktp.safe(i,j) = dfkernel_l(xt.safe[i], xp.safe[j], ll, lv, k);
                dvktp.safe(i,j) = dfkernel_v(xt.safe[i], xp.safe[j], ll, lv, k);

                double dp = dfkernel_p(xt.safe[i], xp.safe[j], ll, lv, k);
                dpktp.safe[j].safe(i,j) = dp;
            }
        }

        // Cholesky for Kpp
        matrix::decompose_cholesky lp;
        if (!lp.decompose(kpp)) {
            error("could not Cholesky decompose pseudo input Kernel matrix");
            return ret;
        }

        matrix::mat2d ilp = lp.lower_inverse();

        // Compute V = Lp^-1 Ktp^T
        matrix::mat2d v(np,nt);
        for (uint_t i : range(np))
        for (uint_t j : range(nt))
        for (uint_t k : range(np)) {
            v.safe(i,j) += ilp.safe(i,k)*ktp.safe(j,k);
        }

        // Compute Lambda
        for (uint_t i : range(nt)) {
            double li = ktt.safe[i];
            for (uint_t j : range(np)) {
                li -= sqr(v.safe(j,i));
            }

            lambda.safe[i] = li;
        }

        // Compute y^tilde
        vec1d ytilde = yt/lambda;

        // Compute M
        matrix::mat2d m(np, np);
        for (uint_t i : range(np))
        for (uint_t j : range(i, np)) {
            double tm = (i == j ? 1.0 : 0.0);
            for (uint_t k : range(nt)) {
                tm += v.safe(i,k)*v.safe(j,k)/lambda.safe[k];
            }
            m.safe(i,j) = m.safe(j,i) = tm;
        }

        // Cholesky for M
        matrix::decompose_cholesky lm;
        if (!lm.decompose(m)) {
            error("could not Cholesky decompose M");
            return ret;
        }

        matrix::mat2d ilm = lm.lower_inverse();

        // Compute Lm^-1 V Lambda^-1
        matrix::mat2d ivl(np,nt);
        for (uint_t i : range(np))
        for (uint_t j : range(nt))
        for (uint_t k : range(np)) {
            ivl.safe(i,j) += ilm.safe(i,k)*v.safe(k,j)/lambda.safe[j];
        }

        // Calculate log likelihood
        if (opts == minimize_function_output::value || opts == minimize_function_output::all) {
            // Compute beta
            vec1d beta = ivl*yt;

            double l0 = total(yt*ytilde);
            double l1 = -total(beta*beta);
            double l2 = total(log(lambda));
            double l3 = 2.0*lm.log_lower_determinant();
            double l4 = nt*log(2.0*dpi);

            vif_check(is_finite(l0), "l0 is invalid: ", l0);
            vif_check(is_finite(l1), "l1 is invalid: ", l1);
            vif_check(is_finite(l2), "l2 is invalid: ", l2);
            vif_check(is_finite(l3), "l3 is invalid: ", l3);
            vif_check(is_finite(l4), "l4 is invalid: ", l4);

            ret[0] = l0 + l1 + l2 + l3 + l4;
        }

        // Calculate derivatives
        if (opts == minimize_function_output::derivatives || opts == minimize_function_output::all) {
            // Compute Q
            matrix::mat2d q(np, np);
            for (uint_t i : range(np))
            for (uint_t j : range(i, np)) {
                double tq = kpp.safe(i,j);
                for (uint_t k : range(nt)) {
                    tq += ktp.safe(k,i)*ktp.safe(k,j)/lambda.safe[k];
                }
                q.safe(i,j) = q.safe(j,i) = tq;
            }

            // Cholesky for Q
            matrix::decompose_cholesky lq;
            if (!lq.decompose(q)) {
                error("could not Cholesky decompose Q");
                return ret;
            }

            matrix::mat2d ilq = lq.lower_inverse();

            // Compute B
            matrix::mat2d b(np,nt);
            for (uint_t i : range(np))
            for (uint_t j : range(nt))
            for (uint_t k : range(np)) {
                b.safe(i,j) += ilq.safe(k,i)*ivl.safe(k,j);
            }

            // Compute b
            wb = b*yt;

            // Compute m
            vec1d ym = transpose(b)*q*wb;

            // Compute diagonal of (Lambda^-1 - B^T Q B)
            vec1d lbqb(nt);
            for (uint_t i : range(nt)) {
                double tl = 1.0/lambda.safe[i];
                for (uint_t j : range(np)) {
                    tl -= sqr(ivl.safe(j,i));
                }
                lbqb.safe[i] = tl;
            }

            // Compute (Q^-1 - Kpp^-1)
            iqkpp = lq.invert() - lp.invert(); // faster than using ilq and ilp

            // Compute Lp^-1,T V
            matrix::mat2d lpv = transpose(ilp)*v;

            // Compute derivatives now...

            // Derivatives for pseudo input positions (dp***)
            for (uint_t ip : range(np)) {
                double dl0 = 0.0; // -(ytilde - m)*dLambda*(ytilde - m)
                double dl3 = 0.0; // tr[(Lambda^-1 - BQB)*dLambda]
                for (uint_t i : range(nt)) {
                    double dlambda1 = -dpktp.safe[ip].safe(i,ip);
                    double dlambda2 = 0.0;
                    for (uint_t j : range(np)) {
                        dlambda2 += dpkpp.safe[ip].safe(ip,j)*lpv.safe(j,i);
                    }

                    double dlambda = 2.0*lpv.safe(ip,i)*(dlambda1 + dlambda2);

                    dl0 -= sqr(ytilde.safe[i] - ym.safe[i])*dlambda;
                    dl3 += lbqb.safe[i]*dlambda;
                }

                double dl1 = 0.0; // -2*(ytilde - m)*dktp*b
                double dl4 = 0.0; // 2*tr[B*dktp]
                for (uint_t i : range(nt)) {
                    dl1 -= (ytilde.safe[i] - ym.safe[i])*dpktp.safe[ip].safe(i,ip);
                    dl4 += b.safe(ip,i)*dpktp.safe[ip].safe(i,ip);
                }
                dl1 *= 2.0*wb.safe[ip];
                dl4 *= 2.0;

                double dl2 = 0.0; // b*dkpp*b
                double dl5 = 0.0; // tr[(Q^-1 - Kpp^-1)*dkpp]
                for (uint_t i : range(np)) {
                    dl2 += dpkpp.safe[ip].safe(ip,i)*wb.safe[i];
                    dl5 += dpkpp.safe[ip].safe(ip,i)*iqkpp.safe(i,ip);
                }
                dl2 *= 2.0*wb.safe[ip];
                dl5 *= 2.0;

                vif_check(is_finite(dl0), "dl0 (position ", ip, ") is invalid: ", dl0);
                vif_check(is_finite(dl1), "dl1 (position ", ip, ") is invalid: ", dl1);
                vif_check(is_finite(dl2), "dl2 (position ", ip, ") is invalid: ", dl2);
                vif_check(is_finite(dl3), "dl3 (position ", ip, ") is invalid: ", dl3);
                vif_check(is_finite(dl4), "dl4 (position ", ip, ") is invalid: ", dl4);
                vif_check(is_finite(dl5), "dl5 (position ", ip, ") is invalid: ", dl5);

                ret[1+ip] = dl0 + dl1 + dl2 + dl3 + dl4 + dl5;
            }

            // Derivative for scale length (dl***)
            {
                double dl0 = 0.0; // -(ytilde - m)*dLambda*(ytilde - m)
                double dl3 = 0.0; // tr[(Lambda^-1 - BQB)*dLambda]
                for (uint_t i : range(nt)) {
                    double dlambda1 = 0.0;
                    double dlambda2 = 0.0;
                    for (uint_t j : range(np)) {
                        double tdl = 0.0;
                        for (uint_t k : range(j, np)) {
                            tdl += (j == k ? 1.0 : 2.0)*dlkpp.safe(j,k)*lpv.safe(k,i);
                        }

                        dlambda1 += tdl*lpv.safe(j,i);
                        dlambda2 -= dlktp.safe(i,j)*lpv.safe(j,i);
                    }

                    dlambda2 *= 2.0;

                    double dlambda = dlambda1 + dlambda2;

                    dl0 -= sqr(ytilde.safe[i] - ym.safe[i])*dlambda;
                    dl3 += lbqb.safe[i]*dlambda;
                }

                double dl1 = 0.0; // -2*(ytilde - m)*dktp*b
                double dl4 = 0.0; // 2*tr[B*dktp]
                for (uint_t i : range(nt))
                for (uint_t j : range(np)) {
                    dl1 -= (ytilde.safe[i] - ym.safe[i])*dlktp.safe(i,j)*wb.safe[j];
                    dl4 += b.safe(j,i)*dlktp.safe(i,j);
                }
                dl1 *= 2.0;
                dl4 *= 2.0;

                double dl2 = 0.0; // b*dkpp*b
                double dl5 = 0.0; // tr[(Q^-1 - Kpp^-1)*dkpp]
                for (uint_t i : range(np))
                for (uint_t j : range(np)) {
                    dl2 += wb.safe[i]*dlkpp.safe(i,j)*wb.safe[j];
                    dl5 += iqkpp.safe(j,i)*dlkpp.safe(i,j);
                }

                vif_check(is_finite(dl0), "dl0 (scale length) is invalid: ", dl0);
                vif_check(is_finite(dl1), "dl1 (scale length) is invalid: ", dl1);
                vif_check(is_finite(dl2), "dl2 (scale length) is invalid: ", dl2);
                vif_check(is_finite(dl3), "dl3 (scale length) is invalid: ", dl3);
                vif_check(is_finite(dl4), "dl4 (scale length) is invalid: ", dl4);
                vif_check(is_finite(dl5), "dl5 (scale length) is invalid: ", dl5);

                ret[1+np+0] = dl0 + dl1 + dl2 + dl3 + dl4 + dl5;
            }

            // Derivative for variance (dv***)
            {
                double dl0 = 0.0; // -(ytilde - m)*dLambda*(ytilde - m)
                double dl3 = 0.0; // tr[(Lambda^-1 - BQB)*dLambda]
                for (uint_t i : range(nt))  {
                    double dlambda0 = dvktt.safe[i];
                    double dlambda1 = 0.0;
                    double dlambda2 = 0.0;
                    for (uint_t j : range(np)) {
                        double tdl = 0.0;
                        for (uint_t k : range(j, np)) {
                            tdl += (j == k ? 1.0 : 2.0)*dvkpp.safe(j,k)*lpv.safe(k,i);
                        }

                        dlambda1 += tdl*lpv.safe(j,i);
                        dlambda2 -= dvktp.safe(i,j)*lpv.safe(j,i);
                    }

                    dlambda2 *= 2.0;

                    double dlambda = dlambda0 + dlambda1 + dlambda2;

                    dl0 -= sqr(ytilde.safe[i] - ym.safe[i])*dlambda;
                    dl3 += lbqb.safe[i]*dlambda;
                }

                double dl1 = 0.0; // -2*(ytilde - m)*dktp*b
                double dl4 = 0.0; // 2*tr[B*dktp]
                for (uint_t i : range(nt))
                for (uint_t j : range(np)) {
                    dl1 -= (ytilde.safe[i] - ym.safe[i])*dvktp.safe(i,j)*wb.safe[j];
                    dl4 += b.safe(j,i)*dvktp.safe(i,j);
                }
                dl1 *= 2.0;
                dl4 *= 2.0;

                double dl2 = 0.0; // b*dkpp*b
                double dl5 = 0.0; // tr[(Q^-1 - Kpp^-1)*dkpp]
                for (uint_t i : range(np))
                for (uint_t j : range(np)) {
                    dl2 += wb.safe[i]*dvkpp.safe(i,j)*wb.safe[j];
                    dl5 += iqkpp.safe(j,i)*dvkpp.safe(i,j);
                }

                vif_check(is_finite(dl0), "dl0 (variance) is invalid: ", dl0);
                vif_check(is_finite(dl1), "dl1 (variance) is invalid: ", dl1);
                vif_check(is_finite(dl2), "dl2 (variance) is invalid: ", dl2);
                vif_check(is_finite(dl3), "dl3 (variance) is invalid: ", dl3);
                vif_check(is_finite(dl4), "dl4 (variance) is invalid: ", dl4);
                vif_check(is_finite(dl5), "dl5 (variance) is invalid: ", dl5);

                ret[1+np+1] = dl0 + dl1 + dl2 + dl3 + dl4 + dl5;
            }

            // Derivative for model noise (ds***)
            {
                double dl0 = 0.0; // -(ytilde - m)*dLambda*(ytilde - m)
                double dl3 = 0.0; // tr[(Lambda^-1 - BQB)*dLambda]
                for (uint_t i : range(nt))  {
                    double dlambda0 = 1.0;
                    double dlambda1 = 0.0;
                    for (uint_t j : range(np)) {
                        dlambda1 += sqr(lpv.safe(j,i));
                    }

                    double dlambda = dlambda0 + dlambda1;

                    dl0 -= sqr(ytilde.safe[i] - ym.safe[i])*dlambda;
                    dl3 += lbqb.safe[i]*dlambda;
                }

                double dl1 = 0.0; // -2*(ytilde - m)*dktp*b
                double dl4 = 0.0; // 2*tr[B*dktp]

                double dl2 = 0.0; // b*dkpp*b
                double dl5 = 0.0; // tr[(Q^-1 - Kpp^-1)*dkpp]
                for (uint_t i : range(np)) {
                    dl2 += sqr(wb.safe[i]);
                    dl5 += iqkpp.safe(i,i);
                }

                vif_check(is_finite(dl0), "dl0 (noise) is invalid: ", dl0);
                vif_check(is_finite(dl1), "dl1 (noise) is invalid: ", dl1);
                vif_check(is_finite(dl2), "dl2 (noise) is invalid: ", dl2);
                vif_check(is_finite(dl3), "dl3 (noise) is invalid: ", dl3);
                vif_check(is_finite(dl4), "dl4 (noise) is invalid: ", dl4);
                vif_check(is_finite(dl5), "dl5 (noise) is invalid: ", dl5);

                ret[1+np+2] = ds*(dl0 + dl1 + dl2 + dl3 + dl4 + dl5);
            }
        }

        return ret;
    };

    double t = now();
    vec1d p0 = rgen(min(xt), max(xt), np);
    append(p0, vec1d{length0, log(var0), log(std0)});
    print("init values: ", p0);

    if (test_evidence) {
        vec1d r0 = get_evidence(p0, minimize_function_output::all);
        for (uint_t i : range(p0)) {
            double dp = 1e-5;
            vec1d p1 = p0;
            p1[i] += dp;
            vec1d r1 = get_evidence(p1, minimize_function_output::value);
            print((r1[0] - r0[0])/dp, ", ", r0[1+i]);
        }
    } else {
        minimize_params opts;
        minimize_result r = minimize_bfgs(opts, p0, get_evidence);
        t = now() - t;
        print("time needed: ", t);
        print("success: ", r.success);
        print("iterations: ", r.niter);
        print("values: ", r.params);
        print("loge: ", -r.value);

        vec1d xp = r.params[_-(np-1)];
        double l = r.params[np+0];
        double lv = r.params[np+1];
        double ls = r.params[np+2];

        matrix::mat2d kps(np,ns);
        matrix::mat2d kss(ns,ns);

        for (uint_t i : range(ns))
        for (uint_t j : range(i, ns)) {
            kss.safe(i,j) = kss.safe(j,i) = fkernel(xs.safe[i], xs.safe[j], l, lv) +
                (i == j ? nugget + exp(2*ls) : 0.0);
        }
        for (uint_t i : range(np))
        for (uint_t j : range(ns)) {
            kps.safe(i,j) = fkernel(xp.safe[i], xs.safe[j], l, lv);
        }

        vec1d m = transpose(kps)*wb;
        matrix::mat2d cov = kss + transpose(kps)*iqkpp*kps;

        matrix::decompose_cholesky d;
        if (!d.decompose(cov)) {
            error("could not decompose covariance matrix! try adding small sigma^2 to diagonal");
            return 1;
        }

        auto seed = make_seed(tseed);

        vec2d ys(n, ns);
        for (uint_t i : range(n)) {
            ys(i,_) = m + d.l*randomn(seed, ns);
        }

        fits::write_table(output_file, "ys", ys, "cov", cov.base, "m", m, "xp", xp,
            "length", l, "amplitude", exp(lv), "noise", exp(ls));
    }

    return 0;
}
