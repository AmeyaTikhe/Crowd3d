// JS projection code for CityStreet dataset.
//
// CityStreet: Multi-View Crowd Counting Dataset
// Copyright (c) 2019
// Qi Zhang, Antoni B. Chan
// City University of Hong Kong

function view1ToView2(u, v, Zw){
	var XYw = Image2World("view1", u, v, Zw);
	var XYi2 = World2Image("view2", XYw[0], XYw[1], Zw);
	return XYi2; 
}

function view1ToView3(u, v, Zw){
	var XYw = Image2World("view1", u, v, Zw);
	var XYi3 = World2Image("view3", XYw[0], XYw[1], Zw);
    	return XYi3; 
}

function view2ToView3(u, v, Zw){
    var XYw = Image2World("view2", u, v, Zw);
    var XYi3 = World2Image("view3", XYw[0], XYw[1], Zw);
    return XYi3; 
}

function view2ToView1(u, v, Zw){
    var XYw = Image2World("view2", u, v, Zw);
    var XYi3 = World2Image("view1", XYw[0], XYw[1], Zw);
    return XYi3; 
}

function view3ToView1(u, v, Zw){
    var XYw = Image2World("view3", u, v, Zw);
    var XYi3 = World2Image("view1", XYw[0], XYw[1], Zw);
    return XYi3; 
}

function view3ToView2(u, v, Zw){
    var XYw = Image2World("view3", u, v, Zw);
    var XYi3 = World2Image("view2", XYw[0], XYw[1], Zw);
    return XYi3; 
}



function find_height(x1, y1, x2, y2){
    var height_range = range(1560, 1980, 5);
    var dist_error = new Array(height_range.length);

    var index = 0;
    var error_num = 0;

    for (var i = 0; i< height_range.length; i++){
    	var height_i = height_range[i];
        var g1 = Image2World("view1", x1, y1, height_i);
        //g1_w = g1[0,1];
        var g2 = Image2World("view2", x2, y2, height_i);
        //g2_w = g2[0,1];
		var gx = g1[0]-g2[0];
        var gy = g1[1]-g2[1];

        var error_i = Math.sqrt(gx*gx + gy*gy);
        if(i==0){
        	index = 0;
        	error_num = error_i;
		}
		else {
        	if(error_num>error_i){
        		error_num = error_i;
        		index = i;
			}
		}
	}
        var height = height_range[index];
        var g3 = view1ToView3(x1, y1, height);

        return [g3[0],g3[1], height];
}

function find_height2(x2, y2, x3, y3){
    var height_range = range(1560, 1980, 5);
    var dist_error = new Array(height_range.length);

    var index = 0;
    var error_num = 0;

    for (var i = 0; i< height_range.length; i++){
        var height_i = height_range[i];
        var g2 = Image2World("view2", x2, y2, height_i);
        //g1_w = g1[0,1];
        var g3 = Image2World("view3", x3, y3, height_i);
        //g2_w = g2[0,1];
        var gx = g2[0]-g3[0];
        var gy = g2[1]-g3[1];

        var error_i = Math.sqrt(gx*gx + gy*gy);
        if(i==0){
            index = 0;
            error_num = error_i;
        }
        else {
            if(error_num>error_i){
                error_num = error_i;
                index = i;
            }
        }
    }
    var height = height_range[index];
    //var g3 = view1ToView3(x1, y1, height);

    //return [g3[0],g3[1], height];
    return height;
}

function find_height_3views(x1, y1, x2, y2, x3, y3){
    var height_range = range(1560, 1980, 5);
    var dist_error = new Array(height_range.length);

    var index = 0;
    var error_num = 0;

    for (var i = 0; i< height_range.length; i++){
        var height_i = height_range[i];
        var g1 = Image2World("view1", x1, y1, height_i);
        //g1_w = g1[0,1];
        var g2 = Image2World("view2", x2, y2, height_i);
        //g2_w = g2[0,1];
        var g3 = Image2World("view3", x3, y3, height_i);

        var gx_avg = (g1[0]+g2[0]+g3[0])/3.0;
        var gy_avg = (g1[1]+g2[1]+g3[1])/3.0;
        
        var gx1 = g1[0] - gx_avg;
        var gy1 = g1[1] - gy_avg;
        var gx2 = g2[0] - gx_avg;
        var gy2 = g2[1] - gy_avg;
        var gx3 = g3[0] - gx_avg;
        var gy3 = g3[1] - gy_avg;

        var error_i = Math.sqrt(gx1*gx1 + gy1*gy1 + gx2*gx2 + gy2*gy2 + gx3*gx3 + gy3*gy3);
        if(i==0){
            index = 0;
            error_num = error_i;
        }
        else {
            if(error_num>error_i){
                error_num = error_i;
                index = i;
            }
        }
    }
    var height = height_range[index];
    //var g3 = view1ToView3(x1, y1, height);

    //return [g3[0],g3[1], height];
    return height;
}


function range(start, edge, step) {
    // If only one number was passed in make it the edge and 0 the start.
    if (arguments.length == 1) {
        edge = start;
        start = 0;
    }

    // Validate the edge and step numbers.
    edge = edge || 0;
    step = step || 1;

    // Create the array of numbers, stopping befor the edge.
    for (var ret = []; (edge - start) * step > 0; start += step) {
        ret.push(start);
    }
    return ret;
}




function camerafile(view) {
	switch(view){
	    case 'view1':
	      this.width = 2704;//768;
	      this.height = 1520;//576;
	      this.ncx = 7.9500000000e+02;
	      this.nfx = 7.5200000000e+02;

	      this.dx = 4.8500000000e-03;
	      this.dy = 4.6500000000e-03;
	      this.dpx = 1/(1.2434e+03);//5.1273271277e-03;
	      this.dpy = 1/(1.2489e+03);//4.6500000000e-03; 

	      this.focal = 1; //5.5549183034e+00;
	      this.kappa1 = -0.2090; //5.1113043639e-03;
              this.kappa2 = 0.0382;

	      this.cx = 1.38455e+3;//3.2422149053e+02;
	      this.cy = 747.846499320945;//2.8256650051e+02;
	      this.sx = 1;//1.0937855397e+00;

	      this.tx = 1275.94012233;//8.2873214225e+02;
	      this.ty = 226.52473842;//-3.1754796051e+03;
	      this.tz = 24230.24027107;//3.5469298547e+04;

	      this.rx = 2.0405458695e+00;
	      this.ry = -8.9337703748e-01;
	      this.rz = -4.3056124791e-01;
	      break;

	    case 'view2':  // view 3
	      this.width = 2704;
	      this.height = 1520;
	      this.ncx = 7.9500000000e+02;
	      this.nfx = 7.5200000000e+02;

	      this.dx = 4.8500000000e-03;
	      this.dy = 4.6500000000e-03;
	      this.dpx = 1/(1.25818922e+03);//5.1273271277e-03;
	      this.dpy = 1/(1.25988567e+03);//4.6500000000e-03; 

	      this.focal = 1; //6.8289664701e+00;
	      this.kappa1 = -0.23977946;//-1.2058925618e-02;
              this.kappa2 = 0.05901853;

	      this.cx = 1.29253760e+03;//3.7284162282e+02;
	      this.cy = 7.70518094e+02; //3.1659953686e+02;
	      this.sx = 1; //1.0749039449e+00;
	      
	      this.tx =  9680.47673912;//2.5602150080e+03;
	      this.ty = 2273.39720387;//-1.4836633370e+03;
	      this.tz = 31436.76306171;//5.1063732312e+04;

	      this.rx = 1.7637554450e+00;
	      this.ry = -6.8268312644e-01;
	      this.rz = -6.3321351894e-02;
	      break;

	    case 'view3': // view 4
	      this.width = 2704;
	      this.height = 1520;
	      this.ncx = 7.9500000000e+02;
	      this.nfx = 7.5200000000e+02;

	      this.dx = 4.8500000000e-03;
	      this.dy = 4.6500000000e-03;
	      this.dpx = 1/(1255.9);//5.1273271277e-03;
	      this.dpy = 1/(1257.2);//4.6500000000e-03; 

	      this.focal = 1;//3.1316979686e+00;
	      this.kappa1 = -0.2489;//3.7880262468e-02;
          this.kappa2 = 0.07147814;

	      this.cx = 1.3741e+3;//3.6105480377e+02;
	      this.cy = 785.7148;//3.0732879992e+02;
	      this.sx = 1;//1.0894268482e+00;
	      
	      this.tx = 3309.55016035;//7.4969800372e+03;
	      this.ty = -4603.90350916;//-1.9402122072e+03;
	      this.tz = 29264.19379322;//2.3838027680e+04;

	      this.rx = 1.8665618542e+00;
	      this.ry = 1.5219705811e-01;
	      this.rz = 4.5968889283e-02;
	      break;
	}
}

function Image2World(view, Xi, Yi, Zw) {
    Xi = Xi;
    Yi = Yi;
    Zw = -Zw;

    switch(view){
	    case "view1":
	       var camerainfo = new camerafile("view1");

           var mR11 = -0.03615359; //cb * cg;
           var mR12 =  0.99596323 ; //cg * sa * sb - ca * sg;
           var mR13 = -0.0821594;//sa * sg + ca * cg * sb;
           var mR21 = -0.34251474 ;//cb * sg;
           var mR22 =  0.06488427; //sa * sb * sg + ca * cg;
           var mR23 =  0.93726927;//ca * sb * sg - cg * sa;
           var mR31 =  0.93881658;//-sb;
           var mR32 =  0.06202646;//cb * sa;
           var mR33 =  0.33878628; //ca * cb;
	       break;

	    case "view2":  // view 3
	       var camerainfo = new camerafile("view2");

           var mR11 = -0.99730688  ; //cb * cg;
           var mR12 =  0.06324747 ; //cg * sa * sb - ca * sg;
           var mR13 =  0.03713147;//sa * sg + ca * cg * sb;
           var mR21 =  0.02439897 ;//cb * sg;
           var mR22 = -0.19132768; //sa * sb * sg + ca * cg;
           var mR23 =  0.98122292;//ca * sb * sg - cg * sa;
           var mR31 =  0.06916415;//-sb;
           var mR32 =  0.97948633;//cb * sa;
           var mR33 =  0.18926923; //ca * cb;
	       break;

	    case "view3": // view 4
	       var camerainfo = new camerafile("view3");

           var mR11 = 0.99545502  ; //cb * cg;
           var mR12 = 0.07578782 ; //cg * sa * sb - ca * sg;
           var mR13 =-0.05766726;//sa * sg + ca * cg * sb;
           var mR21 = 0.0167508 ;//cb * sg;
           var mR22 = 0.45675628; //sa * sb * sg + ca * cg;
           var mR23 = 0.88943415;//ca * sb * sg - cg * sa;
           var mR31 = 0.09374816;//-sb;
           var mR32 = -0.88635766;//cb * sa;
           var mR33 =  0.45341083; //ca * cb;
	       break;
    }
  
    var mDx = camerainfo.dx;
    var mDy = camerainfo.dy;
    var mDpx = camerainfo.dpx;
    var mDpy = camerainfo.dpy;
    var mFocal = camerainfo.focal;
    var mKappa1 = camerainfo.kappa1;
    var mKappa2 = camerainfo.kappa2;
    var mCx = camerainfo.cx;
    var mCy = camerainfo.cy;
    var mSx = camerainfo.sx;
    var mTx = camerainfo.tx;
    var mTy = camerainfo.ty;
    var mTz = camerainfo.tz;
    var mRx = camerainfo.rx;
    var mRy = camerainfo.ry;
    var mRz = camerainfo.rz;

    // world coords image coords, rotation and transition
    var sa = Math.sin(mRx);
    var ca = Math.cos(mRx);
    var sb = Math.sin(mRy);
    var cb = Math.cos(mRy);
    var sg = Math.sin(mRz);
    var cg = Math.cos(mRz);

    // var mR11 = cb * cg;
    // var mR12 = cg * sa * sb - ca * sg;
    // var mR13 = sa * sg + ca * cg * sb;
    // var mR21 = cb * sg;
    // var mR22 = sa * sb * sg + ca * cg;
    // var mR23 = ca * sb * sg - cg * sa;
    // var mR31 = -sb;
    // var mR32 = cb * sa;
    // var mR33 = ca * cb;

    //compute camera position
    var mCposx = -(mTx*mR11 + mTy*mR21 + mTz*mR31);
    var mCposy = -(mTx*mR12 + mTy*mR22 + mTz*mR32);
    var mCposz = -(mTx*mR13 + mTy*mR23 + mTz*mR33);

    // step 1:
    var Xd = mDpx * (Xi - mCx) / mSx;
    var Yd = mDpy * (Yi - mCy);
  
    // step 2:
    var XYu = undistortedToDistortedSensorCoord (Xd, Yd, mKappa1)
    Xu = XYu[0];
    Yu = XYu[1];

    // var distortion_factor = 1 + mKappa1 * (Xd*Xd + Yd*Yd);
    // var Xu = Xd * distortion_factor;
    // var Yu = Yd * distortion_factor;
  
    // step 3:
    var common_denominator = ((mR11 * mR32 - mR12 * mR31) * Yu +
        (mR22 * mR31 - mR21 * mR32) * Xu -
        mFocal * mR11 * mR22 + mFocal * mR12 * mR21);

    var Xw = (((mR12 * mR33 - mR13 * mR32) * Yu +
        (mR23 * mR32 - mR22 * mR33) * Xu -
        mFocal * mR12 * mR23 + mFocal * mR13 * mR22) * Zw +
        (mR12 * mTz - mR32 * mTx) * Yu +
        (mR32 * mTy - mR22 * mTz) * Xu -
        mFocal * mR12 * mTy + mFocal * mR22 * mTx) / common_denominator;

    var Yw = -(((mR11 * mR33 - mR13 * mR31) * Yu +
	    (mR23 * mR31 - mR21 * mR33) * Xu -
	    mFocal * mR11 * mR23 + mFocal * mR13 * mR21) * Zw +
	    (mR11 * mTz - mR31 * mTx) * Yu +
	    (mR31 * mTy - mR21 * mTz) * Xu -
	    mFocal * mR11 * mTy + mFocal * mR21 * mTx) / common_denominator; 

    return [Xw, Yw];

}



function World2Image(view, Xw, Yw, Zw) {
     Zw = -Zw;
     switch(view){
	    case "view1":
	       var camerainfo = new camerafile("view1");

           var mR11 = -0.03615359; //cb * cg;
           var mR12 =  0.99596323 ; //cg * sa * sb - ca * sg;
           var mR13 = -0.0821594;//sa * sg + ca * cg * sb;
           var mR21 = -0.34251474 ;//cb * sg;
           var mR22 =  0.06488427; //sa * sb * sg + ca * cg;
           var mR23 =  0.93726927;//ca * sb * sg - cg * sa;
           var mR31 =  0.93881658;//-sb;
           var mR32 =  0.06202646;//cb * sa;
           var mR33 =  0.33878628; //ca * cb;
	       break;

	    case "view2":  //view 3
	       var camerainfo = new camerafile("view2");

           var mR11 = -0.99588488  ; //cb * cg;
           var mR12 =  0.04888128 ; //cg * sa * sb - ca * sg;
           var mR13 =  0.07631462;//sa * sg + ca * cg * sb;
           var mR21 =  0.06391773 ;//cb * sg;
           var mR22 = -0.21812785; //sa * sb * sg + ca * cg;
           var mR23 =  0.97382481;//ca * sb * sg - cg * sa;
           var mR31 =  0.06424815;//-sb;
           var mR32 =  0.97469527;//cb * sa;
           var mR33 =  0.19541615; //ca * cb;
	       break;

	    case "view3": // view 4
	       var camerainfo = new camerafile("view3");

           var mR11 = 0.99545502  ; //cb * cg;
           var mR12 = 0.07578782 ; //cg * sa * sb - ca * sg;
           var mR13 =-0.05766726;//sa * sg + ca * cg * sb;
           var mR21 = 0.0167508 ;//cb * sg;
           var mR22 = 0.45675628; //sa * sb * sg + ca * cg;
           var mR23 = 0.88943415;//ca * sb * sg - cg * sa;
           var mR31 = 0.09374816;//-sb;
           var mR32 = -0.88635766;//cb * sa;
           var mR33 =  0.45341083; //ca * cb;
	       break;
    }
  
    var mDx = camerainfo.dx;
    var mDy = camerainfo.dy;
    var mDpx = camerainfo.dpx;
    var mDpy = camerainfo.dpy;
    var mFocal = camerainfo.focal;
    var mKappa1 = camerainfo.kappa1;
    var mKappa2 = camerainfo.kappa2;
    var mCx = camerainfo.cx;
    var mCy = camerainfo.cy;
    var mSx = camerainfo.sx;
    var mTx = camerainfo.tx;
    var mTy = camerainfo.ty;
    var mTz = camerainfo.tz;
    var mRx = camerainfo.rx;
    var mRy = camerainfo.ry;
    var mRz = camerainfo.rz;

    // world coordsimage coords, rotation and transition
    var sa = Math.sin(mRx);
    var ca = Math.cos(mRx);
    var sb = Math.sin(mRy);
    var cb = Math.cos(mRy);
    var sg = Math.sin(mRz);
    var cg = Math.cos(mRz);

    //var mR11 = cb * cg;
    //var mR12 = cg * sa * sb - ca * sg;
    //var mR13 = sa * sg + ca * cg * sb;
    //var mR21 = cb * sg;
    //var mR22 = sa * sb * sg + ca * cg;
    //var mR23 = ca * sb * sg - cg * sa;
    //var mR31 = -sb;
    //var mR32 = cb * sa;
    //var mR33 = ca * cb;

    //compute camera position
    var mCposx = -(mTx*mR11 + mTy*mR21 + mTz*mR31);
    var mCposy = -(mTx*mR12 + mTy*mR22 + mTz*mR32);
    var mCposz = -(mTx*mR13 + mTy*mR23 + mTz*mR33);

    // step 1:
    var xc = mR11 * Xw + mR12 * Yw + mR13 * Zw + mTx;
    var yc = mR21 * Xw + mR22 * Yw + mR23 * Zw + mTy;
    var zc = mR31 * Xw + mR32 * Yw + mR33 * Zw + mTz;
  
    // step 2:
    var Xu = mFocal * xc / zc;
    var Yu = mFocal * yc / zc;
  
    // step 3:
    var distortion_factor = 1 + mKappa1 * (Xu*Xu + Yu*Yu);
    var Xd = Xu * distortion_factor;
    var Yd = Yu * distortion_factor;

    // var XYd = undistortedToDistortedSensorCoord (Xu, Yu, mKappa1, mKappa2)
    // Xd = XYd[0];
    // Yd = XYd[1];

    // step 4:
    var Xi = Xd * mSx / mDpx + mCx;
    var Yi = Yd / mDpy + mCy;

    return [Xi, Yi];		  
}

function undistortedToDistortedSensorCoord0 (Xu, Yu, mKappa1, mKappa2)
{
    var  Ru;
    var  a;
    var  b;
    var  c;
    var  d;
    var Xd;
    var Yd;
    var Rd;
    var R0;
    var k=0;
    var e = 1e-10;

    if (((Xu == 0) && (Yu == 0)) || (mKappa1 == 0)&& (mKappa2 == 0))
    {
	    Xd = Xu;
	    Yd = Yu;
    }
    else
    {
	    Ru = Math.sqrt(Xu*Xu + Yu*Yu);
            a = mKappa2*mKappa2;
            b = 2*mKappa1*mKappa2;
	    c = mKappa1*mKappa1+2*mKappa2;
	    d = 2*mKappa1;
            
            Rd = Ru;
            while(k<1000){
                 k++;
                 R0=Rd;
                 
                 fR0 = a*Math.pow(R0,5)+b*Math.pow(R0,4)+c*Math.pow(R0,3)+d*Math.pow(R0,2)+R0-Ru;
                 f_R0 = 5*a*Math.pow(R0,4)+4*b*Math.pow(R0,3)+3*c*R0*R0+2*d*R0+1;
                 Rd = R0-fR0/f_R0;
                
                 if(Math.abs(Rd-R0)<=e){
                     break;           
                 }
            }         
	    
	    lambda = 1+mKappa1*Rd+mKappa2*Rd*Rd;
	    Xd = Xu / lambda;
	    Yd = Yu / lambda;    
    }

    return [Xd, Yd];
}

function undistortedToDistortedSensorCoord (Xu, Yu, mKappa1)
{
    var  Ru;
    var  Rd;
    var  lambda;
    var  c;
    var  d;
    var  Q;
    var  R;
    var  D;
    var  S;
    var  T;
    var  sinT;
    var  cosT;
    var Xd;
    var Yd;

    if (((Xu == 0) && (Yu == 0)) || (mKappa1 == 0))
    {
	    Xd = Xu;
	    Yd = Yu;
    }
	    else
	    {
	    Ru = Math.sqrt(Xu*Xu + Yu*Yu);

	    c = 1.0 / mKappa1;
	    d = -c * Ru;

	    Q = c / 3;
	    R = -d / 2;
	    D = Q*Q*Q + R*R;
	    if (D>=0)
	    {
	      /// * one real root */
	      D = Math.sqrt(D);
	      if (R + D > 0)
	      {
	        S = Math.pow(R + D, 1.0/3.0);
	      }
	      else
	      {
	        S = -Math.pow(-R - D, 1.0/3.0);
	      }

	      if (R - D > 0)
	      {
	        T = Math.pow(R - D, 1.0/3.0);
	      }
	      else
	      {
	        T = -Math.pow(D - R, 1.0/3.0);
	      }

	      Rd = S + T;

	      if (Rd < 0)
	      {
	        Rd = Math.sqrt(-1.0 / (3 * mKappa1));
	      }
	    }
	    else
	    {
	      /* three real roots */
	      D = Math.sqrt(-D);
	      S = Math.pow( Math.sqrt(R*R + D*D) , 1.0/3.0 );
	      T = Math.atan2(D, R) / 3;
	      sinT = Math.sin(T);
	      cosT = Math.cos(T);

	      /* the larger positive root is    2*S*cos(T)                   */
	      /* the smaller positive root is   -S*cos(T) + SQRT(3)*S*sin(T) */
	      /* the negative root is           -S*cos(T) - SQRT(3)*S*sin(T) */

	      Rd = -S * cosT + Math.sqrt(3.0) * S * sinT;  /* use the smaller positive root */
	    }
	    lambda = Rd / Ru;
	    Xd = Xu * lambda;
	    Yd = Yu * lambda;    
    }
    return [Xd, Yd];
}
