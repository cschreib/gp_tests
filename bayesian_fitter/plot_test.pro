d = mrdfits('data.fits', 1, /silent)
r = mrdfits('result.fits', 1, /silent)

nt = n_elements(r.xt)

plot, r.xt, r.m, yr=[min(d.y-d.e) < min(r.m), max(d.y+d.e) > max(r.m)]
oplot, r.xt, r.m + r.s[indgen(nt), indgen(nt)], col='ffff'x, line=1
oplot, r.xt, r.m - r.s[indgen(nt), indgen(nt)], col='ffff'x, line=1

errplot, d.x, d.y-d.e, d.y+d.e
oplot, d.x, d.y, psym=5, col='ff'x
oplot, d.x, d.yp, psym=5, col='ff00'x
oplot, d.xt, d.yt, col='ff00'x

end
