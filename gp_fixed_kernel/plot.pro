p = mrdfits('input.fits', 1, /silent)
r = mrdfits('output.fits', 1, /silent)

plot, p.x, p.y, xr=[min(p.xt), max(p.xt)], yr=[-1.2, 1.2], charsize=2, psym=5, /nodata

i = indgen(n_elements(p.xt))
s = r.cov[i,i]

oplot, p.xt, r.m
oplot, p.xt, r.m - 2*s, line=2
oplot, p.xt, r.m + 2*s, line=2

oplot, p.xt, p.yt, col='ff'x

; mm = mean(r.yt, dim=2)
; sm = stddev(r.yt, dim=2)
; oplot, p.xt, mm, col='ff'x
; oplot, p.xt, mm-2*sm, col='ff'x, line=2
; oplot, p.xt, mm+2*sm, col='ff'x, line=2

; oplot, p.xt, r.yt[*,0], col='ffff'x
; oplot, p.xt, r.yt[*,1], col='ff00'x
; oplot, p.xt, r.yt[*,2], col='ff8800'x

oplot, p.x, p.y, psym=5, symsize=2, col='ff00'x

end
