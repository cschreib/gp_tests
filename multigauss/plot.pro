p = mrdfits('params.fits', 1, /silent)
r = mrdfits('result.fits', /silent)

i = indgen(n_elements(p.x))
s = p.cov[i,i]

plot, p.x, p.m, yr=[min(p.m - 2*s), max(p.m + 2*s)], charsize=2

oplot, p.x, p.m - 2*s, line=1
oplot, p.x, p.m + 2*s, line=1

n = n_elements(r[0,*])
color = color_rainbow(findgen(n)/(n-1.001), table='rbow4')

mm = mean(r, dim=2)
sm = stddev(r, dim=2)
oplot, p.x, mm, col='ff'x
oplot, p.x, mm-2*sm, col='ff'x, line=1
oplot, p.x, mm+2*sm, col='ff'x, line=1

oplot, p.x, r[*,0], col='ffff'x
oplot, p.x, r[*,1], col='ff00'x
oplot, p.x, r[*,2], col='ff8800'x

end
