pro pulses,istart=istart,iend=iend

;  v_18 - start to find error bars on rise/fall times p. 129
;  v_17 - change range of tau from 5-40 to 1-40 p. 119
;  v_16 - with restore of pulses_all_31,32,32
;  v_15 - still trying to fit rise
;  v_14 - now with pulse rise
;  v_12 - print only code=0 for 103 shots
;  v_11 - with ishot iteration and printout at end
;  v_10 - with loaner, from _9 which made upside only


device, decomposed=0
loadct,39
!p.multi=[0,2,2]
!p.color=0
!p.background=255

;restore,'pulses_all_31.dat'
;restore,'pulses_all_32.dat'
restore,'pulses_all_33.dat'

;  make arrays of pulse output for each shot

pulsepeak = fltarr(301)  ;  peak value of pulses averaged over one shot
pulsefall = fltarr(301)  ;  fall time of average pulse
pulserise = fltarr(301)  ;  rise time of average pulses
errrise   = fltarr(301)  ;  std of rise
errfall   = fltarr(301)  ;  std of fall

;  plot sum of all blobs over all radii


;
;
;  iterate over shots
;
;


for inow=istart,iend do begin

ishot=inow

timeplot = fltarr(41)

for t=0,40 do begin
timeplot(t)=2.5*(t-20)
;print,i,time(t)
endfor

!p.multi=0
charsiz=3

window, 1, xpos=2900,ypos=-100,xsiz=1000,ysiz=800
loadct,3

print,ishot
if codenum(ishot) ne 0 then goto,skipshot

plot,timeplot,sumblobs(ishot,*)/numblobs(ishot),ystyle=1,$
	xstyle=1,yrange=[0,7],xtit='time (탎ec)',color=0,thick=2, $
	charsiz=3,charthick=2,ytit='normalized signal',psym=-2
xyouts, 0.14,0.85,shotn(ishot),charsiz=3,/normal,charthick=2
;xyouts, 0.21,0.79,'10 msec',charsiz=3,/normal,charthick=2	
xyouts, 0.15,0.80,fix(numblobs(ishot)),charsiz=3,/normal,charthick=2
xyouts, 0.3,0.80,'blobs',charsiz=3,/normal,charthick=2
plots, [0,0],[0,7]
plots,[-50,50],[1,1]

wait,.5



;
;
;  fit downward pulse
;
;


;  fit points [20,40]

hightime = timeplot(20:40)
highblob = sumblobs(ishot,20:40)/numblobs(ishot)


;window, 2, xpos=2900,ypos=-500,xsiz=1000,ysiz=800
;plot,hightime,highblob,psym=-2

ytest = fltarr(41,21)  ;  test fit with 30 decay times for 21 points
xtest = fltarr(21)     ;  horizontal test fit coordinate


;  make a variety of test fits with various tau to compare with data

for tau=1,40 do begin
for i=0,20 do begin
ytest(tau,i) = (highblob(0)-1.)*exp(-hightime(i)/(tau*1.0))+1.
;oplot,hightime,ytest(tau,*),psym=-2,color=200
;wait,1
endfor
endfor


;

window, 3, xpos=2900,ypos=-100,xsiz=1000,ysiz=800
loadct,3

print,ishot
if codenum(ishot) ne 0 then goto,skipshot

plot,timeplot,sumblobs(ishot,*)/numblobs(ishot),ystyle=1,$
	xstyle=1,yrange=[0,7],xtit='time (탎ec)',color=0,thick=2, $
	charsiz=3,charthick=2,ytit='normalized signal',psym=-2
xyouts, 0.14,0.85,shotn(ishot),charsiz=3,/normal,charthick=2
;xyouts, 0.21,0.79,'10 msec',charsiz=3,/normal,charthick=2	
xyouts, 0.15,0.80,fix(numblobs(ishot)),charsiz=3,/normal,charthick=2
xyouts, 0.3,0.80,'blobs',charsiz=3,/normal,charthick=2
plots, [0,0],[0,7]
plots,[-50,50],[1,1]


for tau=1,40 do begin
oplot,hightime,ytest(tau,*),color=200
endfor

;  find best fit

sumsquares_fall = fltarr(41)  ;  sum of differences for this tau
for tau=1,40 do begin

for time=0,20 do begin
sumsquares_fall(tau) = sumsquares_fall(tau) + ( sumblobs(ishot,time+20)/numblobs(ishot)-ytest(tau,time) )^2
endfor

print,sumsquares_fall(tau)
endfor

minsum = min(sumsquares_fall(1:40))

print,''
print,'minsum',minsum

near = min(abs(sumsquares_fall-minsum),tauhigh)

oplot, hightime,ytest(tauhigh,*),color=100,thick=2


print,'tau post in microsec', tauhigh


; plot only best fit

window, 4, xpos=2900,ypos=100,xsiz=1000,ysiz=800
loadct,3

print,ishot
if codenum(ishot) ne 0 then goto,skipshot

plot,timeplot,sumblobs(ishot,*)/numblobs(ishot),ystyle=1,$
	xstyle=1,yrange=[0,7],xtit='time (탎ec)',color=0,thick=2, $
	charsiz=3,charthick=2,ytit='normalized signal',psym=-2
xyouts, 0.14,0.85,shotn(ishot),charsiz=3,/normal,charthick=2
;xyouts, 0.21,0.79,'10 msec',charsiz=3,/normal,charthick=2	
xyouts, 0.15,0.80,fix(numblobs(ishot)),charsiz=3,/normal,charthick=2
xyouts, 0.3,0.80,'blobs',charsiz=3,/normal,charthick=2
plots, [0,0],[0,7]
plots,[-50,50],[1,1]

oplot,hightime,ytest(tauhigh,*),color=200, thick=4

xyouts, 0.07,0.75,tauhigh,charsiz=3,/normal,charthick=2,color=200
xyouts, 0.27,0.75,'탎ec fall time',charsiz=3,/normal,charthick=2,color=200


;
;
;
;  now do rise time
;
;
;

;  fit points [0,20]

lowtime = timeplot(0:20)
lowblob = sumblobs(ishot,0:20)/numblobs(ishot)


ytest2 = fltarr(41,21)  ;  test fit with 40 rise times for 21 points
xtest2 = fltarr(21)     ;  horizontal test fit coordinate


;  make a variety of test fits with various tau to compare with data

for tau=1,40 do begin
for i=20,0,-1 do begin  ;  count down from 20
ytest2(tau,i) = (lowblob(20)-1.)*exp(lowtime(i)/(tau*1.0))+1.
;oplot,hightime,ytest(tau,*),psym=-2,color=200
;wait,1
endfor
endfor


;

window, 5, xpos=2900,ypos=-300,xsiz=1000,ysiz=800
loadct,3

print,ishot
if codenum(ishot) ne 0 then goto,skipshot

plot,timeplot,sumblobs(ishot,*)/numblobs(ishot),ystyle=1,$
	xstyle=1,yrange=[0,7],xtit='time (탎ec)',color=0,thick=2, $
	charsiz=3,charthick=2,ytit='normalized signal',psym=-2
xyouts, 0.14,0.85,shotn(ishot),charsiz=3,/normal,charthick=2
;xyouts, 0.21,0.79,'10 msec',charsiz=3,/normal,charthick=2	
xyouts, 0.15,0.80,fix(numblobs(ishot)),charsiz=3,/normal,charthick=2
xyouts, 0.3,0.80,'blobs',charsiz=3,/normal,charthick=2
plots, [0,0],[0,7]
plots,[-50,50],[1,1]


for tau=1,40 do begin
oplot,lowtime,ytest2(tau,*),color=200
endfor

;  find best fit

sumsquares_rise = fltarr(41)  ;  sum of differences for this tau
for tau=1,40 do begin

for time=0,20 do begin
sumsquares_rise(tau) = sumsquares_rise(tau) + ( sumblobs(ishot,time)/numblobs(ishot)-ytest2(tau,time) )^2
endfor

print,sumsquares_rise(tau)
endfor

minsum_rise = min(sumsquares_rise(1:40))

print,''
print,'minsum_rise',minsum_rise

near = min(abs(sumsquares_rise-minsum_rise),taulow)

oplot, lowtime,ytest2(taulow,*),color=100,thick=2


print,'tau post in microsec', taulow


; plot only best fit

window, 6, xpos=1500,ypos=-00,xsiz=1000,ysiz=800
loadct,3

print,ishot
if codenum(ishot) ne 0 then goto,skipshot

plot,timeplot,sumblobs(ishot,*)/numblobs(ishot),ystyle=1,$
	xstyle=1,yrange=[0,7],xtit='time (탎ec)',color=0,thick=2, $
	charsiz=3,charthick=2,ytit='normalized signal',psym=-2
xyouts, 0.14,0.85,shotn(ishot),charsiz=3,/normal,charthick=2
;xyouts, 0.21,0.79,'10 msec',charsiz=3,/normal,charthick=2	
xyouts, 0.15,0.80,fix(numblobs(ishot)),charsiz=3,/normal,charthick=2
xyouts, 0.3,0.80,'blobs',charsiz=3,/normal,charthick=2
plots, [0,0],[0,7]
plots,[-50,50],[1,1]

oplot,lowtime,ytest2(taulow,*),color=200, thick=4
oplot,hightime,ytest(tauhigh,*),color=200, thick=4

xyouts, 0.07,0.75,taulow,charsiz=3,/normal,charthick=2,color=200
xyouts, 0.27,0.75,'탎ec rise time',charsiz=3,/normal,charthick=2,color=200

xyouts, 0.07,0.70,tauhigh,charsiz=3,/normal,charthick=2,color=200
xyouts, 0.27,0.70,'탎ec fall time',charsiz=3,/normal,charthick=2,color=200

xyouts, .75,0.85,ishot,/normal,charsiz=3,charthick=2

wait,0.5

pulsepeak(ishot) = sumblobs(ishot,20)/numblobs(ishot)
pulsefall(ishot) = tauhigh  ; fall time of pulse
pulserise(ishot) = taulow   ; rise time of pulse


skipshot:

window, 7, xpos=500,ypos=100,xsiz=800,ysiz=600
plot, sumsquares_fall
oplot,sumsquares_rise, color=200

errfall(ishot) = sqrt(sumsquares_fall(tauhigh)/20.)
errrise(ishot) = sqrt(sumsquares_rise(taulow)/20.)

xyouts, 0.3,0.8,errfall(ishot),/normal,charsiz=3
xyouts, 0.3,0.7,errrise(ishot),/normal,charsiz=3,color=200

endfor  ; end of iteration over shot








;  print out results

print,''
tab=string(9b)

for i=istart,iend do begin
if codenum(i) ne 0 then goto,skipshot2
print, i,tab,shotn(i),tab,numblobs(i),tab,pulsepeak(i),tab,pulserise(i),tab,pulsefall(i),tab,errrise(i),tab,errfall(i)
skipshot2:
endfor
stop
end


