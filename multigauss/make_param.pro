nx = 100
x = rgen(0,10,nx)
m = x*0
cov = fltarr(nx,nx)

seed = 42

l = replicate(1.0, nx)
l[nx/2] *= 0.9

for i=0, nx-1 do for j=i, nx-1 do begin
    ; d2 = (x[i] - x[j])^2/(1.0 + 0.1*(2.0 + cos(x[i]/10*!dpi*3.0)))^2
    ; d2 = (x[i] - x[j])^2/0.1^3
    d2 = ((x[i] - x[j])/(l[i]*l[j]))^2
    if d2 lt 110 then cov[i,j] = exp(-0.5d0*d2) else cov[i,j] = 0.0
    cov[j,i] = cov[i,j]

    ; if i eq j then cov[i,i] += 0.1*(1.0 + (x[i]/float(nx)))
    if i eq j then cov[i,i] += 0.001
endfor

mwrfits, /create, {x:x, m:m, cov:cov}, 'params.fits'

end
