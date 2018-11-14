p = mrdfits('input.fits', 1, /silent)
r = mrdfits('output.fits', 1, /silent)
rf = mrdfits('output_fgp.fits', 1, /silent)

plot, p.x, p.y, xr=[min(p.xt), max(p.xt)], yr=[1.2*min(p.y), 1.2*max(p.y)], charsize=2, psym=5, /nodata

i = indgen(n_elements(p.xt))
s = sqrt(r.cov[i,i])
sf = sqrt(rf.cov[i,i])

oplot, p.xt, r.m
oplot, p.xt, r.m - 2*s, line=2
oplot, p.xt, r.m + 2*s, line=2

oplot, p.xt, rf.m, col='ffff'x
oplot, p.xt, rf.m - 2*sf, line=2, col='ffff'x
oplot, p.xt, rf.m + 2*sf, line=2, col='ffff'x

oplot, p.xt, p.yt, col='ff'x

; mm = mean(r.yt, dim=2)
; sm = stddev(r.yt, dim=2)
; oplot, p.xt, mm, col='ff'x
; oplot, p.xt, mm-2*sm, col='ff'x, line=2
; oplot, p.xt, mm+2*sm, col='ff'x, line=2

; oplot, p.xt, r.yt[*,0], col='ffff'x
; oplot, p.xt, r.yt[*,1], col='ff00'x
; oplot, p.xt, r.yt[*,2], col='ff8800'x

; oplot, p.x, p.y, psym=5, symsize=0.5, col='ff00'x

for i=0, n_elements(r.xp)-1 do begin
    oplot, r.xp[i]+[0,0], interpol(!y.crange, [0,1], [0.05,0.1]), col='ff8800'x
endfor

end
