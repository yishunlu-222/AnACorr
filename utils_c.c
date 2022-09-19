#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#define PI 3.14159265358979323846f

//cc -fPIC -shared -o finding_faces.so finding_faces.c

int cleanup(double **ppMat, int m)
{
    int ret = 0;
    if (ppMat) {
        printf("\n----- FROM C: free\n");
        for (int i = 0; i < m; i++) {
            free(ppMat[i]);
            ret++;
            ppMat[i] = NULL;
        }
        free(ppMat);
    }
    return ++ret;
}


double **init(double **ppMat, int m, int n)
{
    const double factor = 7.0;
    printf("\n----- FROM C: Multiplying input matrix by: %.3f\n", factor);
	printf("the number of row is %d \n", sizeof(ppMat) / sizeof(ppMat[0]));
	printf("the number of colums is %d \n", sizeof(ppMat[0]) / sizeof(ppMat[0][0]));
    double **ret = malloc(m * sizeof(double*));
    for (int i = 0; i < m; i++) {
        ret[i] = malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            ret[i][j] = ppMat[i][j] * factor;
        }
    }
    return ret;
}

char* db_testing(int8_t*** label_list, int32_t* coord) {
	char* first = (char *)malloc(sizeof(char));
	int x = coord[2];
	int y = coord[1];
	int z = coord[0];
	//printf("",)
	printf("the coord is %p, %p, %p \n", coord[0], coord[1], coord[2]);
	printf("the coord is %d, %d, %d \n", coord[0], coord[1], coord[2]);
	printf("the coord is %d \n", label_list[0]);
	printf("the coord is %d \n", label_list[0][0]);
	printf("the coord is %d \n", label_list[coord[0]][coord[1]][coord[2]]);
	//printf("the coord is %d \n", label_list[93 * max_x * max_y + 508 * max_y + 590]);


	//printf("this point is %d \n",
	//	label_list[ max_x * max_y*max_z]);
	first = &label_list[z][y][x];

	return first;

}


int* testing(int8_t* label_list, int32_t * coord, int max_z, int
	max_y, int max_x) {
	int* first = malloc(sizeof(double));

	//printf("",)
	printf("the coord is %p, %p, %p \n",coord[0], coord[1], coord[2]);
	printf("the coord is %d, %d, %d \n", coord[0], coord[1], coord[2]);
	printf("the coord is %d, %d, %d \n", max_z, max_y, max_x);
	printf("the coord is %d \n", label_list[0]);
	printf("the coord is %d \n", label_list[93 * max_x * max_y + 508 * max_y + 590]);


	//printf("this point is %d \n",
	//	label_list[ max_x * max_y*max_z]);
	//first = &label_list[z][y][x];

	return first;

}

void testingString(char ** strings, int n ) {
	for (int i = 0; i < n; i++) {
		printf("the content is  %s \n", strings[i]);
	}


}

void print_matrix(double* v, size_t n, size_t p)
{
	//for (size_t i = 0; i < n; i++) {
	//	for (size_t j = 0; j < p; j++) {
	//		printf("%f ", v[i * n + j]);
	//	}
	//	printf("\n");
	//}
	printf("\n");
	printf("the coord is %f, %f, %f \n", v[0], v[1], v[2]);
	printf("the coord is %p, %p, %p \n", v[0], v[1], v[2]);
}


double * dials_2_theta(double s1[3], bool pathLength1) {
	double rotated_s1[3];
	double theta;
	double* theta_ptr = malloc(sizeof(double));
	if (pathLength1) {
		
		for (int i; i = 0; i++) {
			rotated_s1[i] = -s1[i];
		}
	}
	else {
		for (int i; i = 0; i++) {
			rotated_s1[i] = s1[i];
		}
	}


	if (rotated_s1[2] == 0) {
		theta = atan(rotated_s1[1] /
			(-sqrt(pow(rotated_s1[0], 2) + pow(rotated_s1[2], 2))) + 0.001);

	}
	else {
		if (rotated_s1[2] < 0) {
			theta =  atan(rotated_s1[1] /
				sqrt(pow(rotated_s1[0], 2) + pow(rotated_s1[2], 2)));
		}
		else {
			if (rotated_s1[1] > 0) {
				theta = PI - atan(rotated_s1[1] /
					sqrt(pow(rotated_s1[0], 2) + pow(rotated_s1[2], 2)));
			}
			else {
				theta = - PI - atan(rotated_s1[1] /
					sqrt(pow(rotated_s1[0], 2) + pow(rotated_s1[2], 2)));
			}
			}

		}
	
	theta_ptr = &theta;
	return theta_ptr;
}

double* cal_coord_2(int face, char*** label_list, double theta, double phi, int* coord, int* LabelShape) {
	
	double z = (double) coord[0], y = (double) coord[1], x = (double) coord[2];
	double* numbers = (double*)malloc(4 * sizeof(double));
	int z_max = LabelShape[0] - 1 , y_max = LabelShape[1] - 1, x_max = LabelShape[2] - 1;
	int i, j, k;
	double cr_length = 0, li_length = 0, lo_length = 0, bu_length = 0;
	int cr_num = 0, li_num = 0, lo_num = 0, bu_num = 0;
	double new_x, new_y, new_z,  increment = 0.0;
	double cls_start_coord_x = x, cls_start_coord_y = y, cls_start_coord_z = z;
	double cls_end_coord_x = x, cls_end_coord_y = y, cls_end_coord_z = z;
	int cls_start_increment = 0, cls_end_increment = 0;

	if (face == 6){
		double increment_ratio_x = -1.0;
		double increment_ratio_y = tan(theta) / cos(phi);
		double increment_ratio_z = tan(phi);
		char cls = 3, flag = 0;
		//printf("the current ratio is [%f,%f,%f] \n", increment_ratio_x, increment_ratio_y, increment_ratio_z);
		
		for (i = 0; i < x; i++) {
			if (theta > 0) {
				new_x = floor(x + increment * increment_ratio_x);
				new_y = floor(y - increment * increment_ratio_y);
				new_z = floor(z + increment * increment_ratio_z);

			}
			else {

				new_x = round(x + increment * increment_ratio_x);	
				new_y = round(y - increment * increment_ratio_y);
				new_z = round(z + increment * increment_ratio_z);

			}
			increment += 1;
			
			cls_end_increment = increment;

			printf("the class is are %d \n", label_list[(int)new_z][(int)new_y][(int)new_x]);
			if (label_list[(int)new_z][(int)new_y][(int)new_x] == cls) {
				if (label_list[(int)new_z][(int)new_y][(int)new_x] == 0) {
					continue;
				}
			}
			else {
				if (cls == 3){
					printf("increments are %d %d \n", cls_end_increment, cls_start_increment);
					cr_num += cls_end_increment - cls_start_increment;
				}
				else if (cls == 1) {
					li_num += cls_end_increment - cls_start_increment;
				}
				else if (cls == 2) {
					lo_num += cls_end_increment - cls_start_increment;
				}
				else if (cls == 4) {
					bu_num += cls_end_increment - cls_start_increment;
				}
				else {

				}
				cls = label_list[(int)new_z][(int)new_y][(int)new_x];
				cls_start_increment = increment;
			}
			
		}
	}
	printf("the cr length is [%d] \n", cr_num);
	printf("the li length is [%d] \n", li_num);
	printf("the lo length is [%d] \n", lo_num);
	printf("the bu length is [%d] \n", bu_num);
	return numbers;
}




int find_face_ray_tracing( double theta, double phi, int* coord, int z_max, int
    y_max, int x_max) {
    /*  'FRONTZY' = 1;
    *   'LEYX' = 2 ;
    *   'RIYX' = 3;
        'TOPZX' = 4;
        'BOTZX' = 5;
        "BACKZY" = 6 ;

    */

	//int z_max = shape[0], y_max = shape[1], x_max = shape[2];
	x_max -= 1;
	z_max -= 1;
	y_max -= 1;
	int z = coord[0], y = coord[1], x = coord[2];
    //printf("the coord is %d, %d, %d \n", z, y, x);
    int face;
	double side;
	//printf("coord is at address %p \n", coord);
	//printf("coord is at address %d \n", coord[0]);
	//printf("coord is at address %d \n", coord[1]);
	//printf("coord is at address %d \n", coord[2]);
    if (fabs(theta) < PI / 2) {

		double theta_up = atan((y - 0) / (x - 0 + 0.001));
		double theta_down = -atan((y_max - y) / (x - 0 + 0.001));  // negative
		double phi_right = atan((z_max - z) / (x - 0 + 0.001));
		double phi_left = -atan((z - 0) / (x - 0 + 0.001));  // negative
		double omega = atan(tan(theta) * cos(phi));
        if (omega > theta_up) {
            // at this case, theta is positive,
            // normally the most cases for theta > theta_up, the ray passes the top ZX plane
            // if the phis are smaller than both edge limits
            // the ray only goes through right / left plane when the  reflection coordinate is too close to the  right / left plane
			side = (y - 0) * sin(fabs(phi)) / tan(theta);  // the length of rotation is the projected length on x
            
            if (side > (z - 0) && phi < phi_left) {

                face = 2;  //face = 'LEYX'

            }
            else if (side > (z_max - z) && phi > phi_right) {

                face = 3;  //face = 'RIYX'

            }  
            else {

				face = 4; //face = 'TOPZX'

            }
            
        }

        else if(omega < theta_down) {
            side = (y_max - y) * sin(fabs(phi)) / tan(-theta);

            if ( side > (z - 0) && phi < phi_left) {

                 face = 2;  //face = 'LEYX'

            }
            else if (side > (z_max - z) && phi > phi_right) {

                face = 3;  //face = 'RIYX'

            }
            else {

				face = 5; //face = 'BOTZX'

            }
             
        }
		else if (phi > phi_right) {

			face = 3; // face = 'RIYX'

		}
		else if (phi < phi_left) {

			face = 2; // face= 'LEYX'

		}
		else {

			face = 6; // face = "BACKZY"

		}
		// ray passes through the back plane

    }


	else{

		// theta is larger than 90 degree or smaller than - 90
		double theta_up = atan((y - 0) / (x_max - x + 0.001));
		double theta_down = atan((y_max - y) / (x_max - x + 0.001)); // negative
		double phi_left = atan((z_max - z) / (x_max - x + 0.001)); // it is the reverse of the top phi_left
		double phi_right = -atan((z - 0) / (x_max - x + 0.001));  // negative

		if ((PI - theta) > theta_up && theta > 0) {
			// at this case, theta is positive,
			//normally the most cases for theta > theta_up, the ray passes the top ZX plane
			//if the phis are smaller than both edge limits
			// the ray only goes through right / left plane when the  reflection coordinate is too close to the  right / left plane
			side = (y - 0) *  sin(fabs(phi)) /  fabs( tan(theta));

			if (side > (z - 0) && -phi < phi_right) {

				face = 2;  // face = 'LEYX'

				} 
			else if (side > (z_max - z) && -phi > phi_left) {

				face = 3; //face = 'RIYX'

			}
			else {

				face = 6;	//		face = 'TOPZX'

			}

		
		}

		else if (theta > theta_down - PI && theta <= 0) {
			side = (y_max - y) * sin(fabs(phi)) / fabs(tan(-theta));

			if (side > (z - 0) && -phi < phi_right) {

				face = 2;  // face = 'LEYX'
			
			}
			else if (side > (z_max - z) && -phi > phi_left){

				face = 3;  //face = 'RIYX'

			}
			else {

				face = 5; // face = 'BOTZX'

			}

		}

		else if (-phi < phi_right) {

			face = 2; //face = 'LEYX'
		
		}
		// when the code goes to this line, it means the theta is within the limits
		
		else if (-phi > phi_left) {
			face = 3; //face = 'RIYX'
		}
	
		else {
			// ray passes through the back plane
			face = 1 ;  //face = 'FRONTZY'
		}
			
	}

	return  face;
}



double absorption_factor(double *numbers, double *coefficients, double pixel_size) {
	double abs1;
	double mu_li = coefficients[0], mu_lo = coefficients[1];
	double mu_cr = coefficients[2], mu_bu = coefficients[3];

	//	if len(numbers) == 8:
	//li_l_1, lo_l_1, cr_l_1, bu_l_1, li_l_2, lo_l_2, cr_l_2, bu_l_2 = numbers
	//	else:
	//li_l_2, lo_l_2, cr_l_2, bu_l_2 = numbers
	//	li_l_1, lo_l_1, cr_l_1, bu_l_1 = 0, 0, 0, 0
	//double li_l_1 = numbers[0], lo_l_1 = numbers[1], cr_l_1 = numbers[2], bu_l_1 = numbers[3]
	//double li_l_2 = numbers[4], lo_l_2 = numbers[5], cr_l_2 = numbers[6], bu_l_2 = numbers[7]
	
	double  li_l_1 = numbers[0] + numbers[4], 
			lo_l_1 = numbers[1] + numbers[5],
			cr_l_1 = numbers[2] + numbers[6], 
			bu_l_1 = numbers[3] + numbers[7];

	abs1 = exp(-((mu_li * (li_l_1 - 0.5) +
		mu_lo * (lo_l_1 - 0.5) +
		mu_cr * (cr_l_1 - 0.5) +
		mu_bu * (bu_l_1 - 0.5)) * pixel_size
		));



	return abs1;
}

double* cal_paths_plus_c_fraction(int** path_ray, int* posi, int* classes,
	int* coord, int SizeClasses, int posiLen, int path_rayLen) {
	/*
	the string labels are convert into int
	'va' ==> 0
	'li' ==> 11
	'lo' ==> 22
	'cr' ==> 33
	'bu' ==> 44

	*/
	int i, j, k;
	double* numbers = (double*)malloc(4 * sizeof(double));
	double  cr_l_2 = 0,
		lo_l_2 = 0,
		li_l_2 = 0,
		bu_l_2 = 0;
	//double x_cr, y_cr, z_cr;
	//double x_li, y_li, z_li;
	//double x_lo, y_lo, z_lo;
	//double x_bu, y_bu, z_bu;
	//double cr_l_2_total, li_l_2_total, lo_l_2_total, bu_l_2_total;
	double total_length = sqrt((
		path_ray[path_rayLen-1][1] - path_ray[0][1]) * (path_ray[path_rayLen - 1][1] - path_ray[0][1]) +
		(path_ray[path_rayLen - 1][0] - path_ray[0][0]) * (path_ray[path_rayLen - 1][0] - path_ray[0][0]) +
		(path_ray[path_rayLen - 1][2] - path_ray[0][2]) * (path_ray[path_rayLen - 1][2] - path_ray[0][2]));

	for (j = 0; j < SizeClasses; j++) {
		if (classes[j] == 33) {

			if (j < posiLen - 1) {
				cr_l_2 += total_length * ((posi[j + 1] - posi[j]) / (float)path_rayLen );

			}
				
			else {
				cr_l_2 += total_length * ((path_rayLen  - posi[j]) /(float)path_rayLen );
			}


		}

		else if (classes[j] == 11) {
			if (j < posiLen - 1) {
				li_l_2 += total_length * ((posi[j + 1] - posi[j]) /(float)path_rayLen );
				}
					
			else {
				li_l_2 += total_length * ((path_rayLen  - posi[j]) /(float)path_rayLen );
			}
		}

		else if(classes[j] == 22) {
			if (j < posiLen - 1){
				lo_l_2 += total_length * ((posi[j + 1] - posi[j]) / (float)path_rayLen );
			}
			else {
				lo_l_2 += total_length * ((path_rayLen - posi[j]) / (float)path_rayLen );
			}
		}

		else if(classes[j] == 44) {
			if (j < posiLen - 1) {
				bu_l_2 += total_length * ((posi[j + 1] - posi[j]) /(float)path_rayLen  );
			}
			else {
				bu_l_2 += total_length * ((path_rayLen  - posi[j]) /(float)path_rayLen  );
			}
				
		}

		else {
		}
	}

	//}
	//printf("the li_l_2 is %d \n", posi[j + 1]);
	//printf("the li_l_2 is %d \n", posi[j]);
	//printf("the li_l_2 is %f \n", li_l_2);
	//printf("the lo_l_2 is %f \n", lo_l_2);
	//printf("the cr_l_2 is %f \n", cr_l_2);
	//printf("the bu_l_2 is %f \n", bu_l_2);
	numbers[0] = li_l_2;
	numbers[1] = lo_l_2;
	numbers[2] = cr_l_2;
	numbers[3] = bu_l_2;
	return numbers;
}


double* cal_paths_plus_c_fraction_singlePtr(int* path_ray, int* posi, int* classes,
	int* coord, int SizeClasses, int posiLen, int* path_shape) {
	/*
	the string labels are convert into int
	'va' ==> 0
	'li' ==> 11
	'lo' ==> 22
	'cr' ==> 33
	'bu' ==> 44

	*/
	int i, j, k;
	int path_maxY = path_shape[0] , path_maxX = path_shape[1];
	double* numbers = malloc(4 * sizeof(double));
	double  cr_l_2 = 0,
			lo_l_2 = 0,
			li_l_2 = 0,
			bu_l_2 = 0;
	//double x_cr, y_cr, z_cr;
	//double x_li, y_li, z_li;
	//double x_lo, y_lo, z_lo;
	//double x_bu, y_bu, z_bu;
	//double cr_l_2_total, li_l_2_total, lo_l_2_total, bu_l_2_total;
	double total_length = sqrt(
		(path_ray[(path_maxY - 1)*path_maxX + 1 ] - path_ray[1]) * (path_ray[(path_maxY - 1) * path_maxX + 1] - path_ray[1]) +
		(path_ray[(path_maxY - 1) * path_maxX + 0] - path_ray[0]) * (path_ray[(path_maxY - 1) * path_maxX + 0] - path_ray[0]) +
		(path_ray[(path_maxY - 1) * path_maxX + 2] - path_ray[2]) * (path_ray[(path_maxY - 1) * path_maxX + 2] - path_ray[2]));

	for (j = 0; j < SizeClasses; j++) {
		if (classes[j] == 33) {

			if (j < posiLen - 1) {
				cr_l_2 += total_length * ((posi[j + 1] - posi[j]) / (float)path_maxY);

			}

			else {
				cr_l_2 += total_length * ((path_maxY - posi[j]) / (float)path_maxY);
			}


		}

		else if (classes[j] == 11) {
			if (j < posiLen - 1) {
				li_l_2 += total_length * ((posi[j + 1] - posi[j]) / (float)path_maxY);
			}

			else {
				li_l_2 += total_length * ((path_maxY - posi[j]) / (float)path_maxY);
			}
		}

		else if (classes[j] == 22) {
			if (j < posiLen - 1) {
				lo_l_2 += total_length * ((posi[j + 1] - posi[j]) / (float)path_maxY);
			}
			else {
				lo_l_2 += total_length * ((path_maxY - posi[j]) / (float)path_maxY);
			}
		}

		else if (classes[j] == 44) {
			if (j < posiLen - 1) {
				bu_l_2 += total_length * ((posi[j + 1] - posi[j]) / (float)path_maxY);
			}
			else {
				bu_l_2 += total_length * ((path_maxY - posi[j]) / (float)path_maxY);
			}

		}

		else {
		}
	}

	//}
	//printf("the li_l_2 is %d \n", posi[j + 1]);
	//printf("the li_l_2 is %d \n", posi[j]);
	//printf("the li_l_2 is %f \n", li_l_2);
	//printf("the lo_l_2 is %f \n", lo_l_2);
	//printf("the cr_l_2 is %f \n", cr_l_2);
	//printf("the bu_l_2 is %f \n", bu_l_2);
	numbers[0] = li_l_2;
	numbers[1] = lo_l_2;
	numbers[2] = cr_l_2;
	numbers[3] = bu_l_2;
	return numbers;
}



double* cal_paths_plus_c_pt(int** path_ray, int * posi, int* classes,
	int* coord,int SizeClasses, int posiLen, int path_rayLen) {
	/*  Pythagorean theorem
	the string labels are convert into int
	'va' ==> 0
	'li' ==> 11
	'lo' ==> 22
	'cr' ==> 33
	'bu' ==> 44

	*/
	int i, j, k;
	//printf("the the coord is %d %d %d \n", coord[0], coord[1], coord[2]);
	
	//for (i = 0; i < 3; i++) {
	//	printf("the the the last one is %d \n", path_ray[path_rayLen-1][i]);
	//}
	//for (i = 0; i < posiLen; i++) {
	//	printf("the the posi is %d \n", posi[i]);
	//}
	//int posiL = sizeof(posi) / sizeof(posi[0]);
	//printf("the the posi length is %d \n", posiL);
	//int posiL = sizeof(posi) / sizeof(posi[0]);
	//int path_rayLen = sizeof(path_ray) / sizeof(path_ray[0]);
	//printf("the the posi length is %d \n", SizeClasses);
	//printf("the the posi length is %d \n", posiLen);
	//printf("the the posi length is %d \n", path_rayLen);
	//printf("the omega is %f", fabs(omega));

	double* numbers =malloc(4 * sizeof(double));
	double  cr_l_2 = 0,
			lo_l_2 = 0,
			li_l_2 = 0,
			bu_l_2 = 0; 
	double x_cr, y_cr, z_cr;
	double x_li, y_li, z_li;
	double x_lo, y_lo, z_lo;
	double x_bu, y_bu, z_bu;
	double cr_l_2_total, li_l_2_total, lo_l_2_total, bu_l_2_total;

	for (j = 0; j < SizeClasses; j++) {
		if (classes[j] == 33) {
			if (j < posiLen - 1) {
				x_cr = abs(path_ray[posi[j + 1]][2] - path_ray[posi[j]][2]);
				z_cr = abs(path_ray[posi[j + 1]][0] - path_ray[posi[j]][0]);
				y_cr = abs(path_ray[posi[j + 1]][1] - path_ray[posi[j]][1]);
			}
			else {
				x_cr = abs(path_ray[path_rayLen-1][2] - path_ray[posi[j]][2]);
				z_cr = abs(path_ray[path_rayLen-1][0] - path_ray[posi[j]][0]);
				y_cr = abs(path_ray[path_rayLen-1][1] - path_ray[posi[j]][1]);
			}
			cr_l_2_total = sqrt((x_cr + 0.5) * (x_cr + 0.5)
				+ (z_cr + 0.5) * (z_cr + 0.5) +
				(y_cr + 0.5) * (y_cr + 0.5));
			cr_l_2 += cr_l_2_total;
		}
		else if (classes[j] == 11) {
			if (j < posiLen - 1) {
				x_li = abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j]][2]);
				z_li = abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j]][0]);
				y_li = abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j]][1]);

			}
			else {
				x_li = abs(path_ray[path_rayLen-1][2] - path_ray[posi[j]][2]);
				z_li = abs(path_ray[path_rayLen-1][0] - path_ray[posi[j]][0]);
				y_li = abs(path_ray[path_rayLen-1][1] - path_ray[posi[j]][1]);
			}
			li_l_2_total = sqrt(x_li * x_li
				+ z_li * z_li
				+ y_li * y_li);
			li_l_2 += li_l_2_total;
		}
		else if (classes[j] == 22) {
			if (j < posiLen - 1) {
				x_lo = abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j]][2]);
				z_lo = abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j]][0]);
				y_lo = abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j]][1]);
			}
			else {
				x_lo = abs(path_ray[path_rayLen-1][2] - path_ray[posi[j]][2]);
				z_lo = abs(path_ray[path_rayLen-1][0] - path_ray[posi[j]][0]);
				y_lo = abs(path_ray[path_rayLen-1][1] - path_ray[posi[j]][1]);

			}
			lo_l_2_total = sqrt(x_lo * x_lo
				+ z_lo * z_lo
				+ y_lo * y_lo);
			lo_l_2 += lo_l_2_total;
		}
		else if (classes[j] == 44) {
			if (j < posiLen - 1) {
				x_bu = abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j]][2]);
				z_bu = abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j]][0]);
				y_bu = abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j]][1]);

			}

			else {
				x_bu = abs(path_ray[path_rayLen-1][2] - path_ray[posi[j]][2]);
				z_bu = abs(path_ray[path_rayLen-1][0] - path_ray[posi[j]][0]);
				y_bu = abs(path_ray[path_rayLen-1][1] - path_ray[posi[j]][1]);
			}

			bu_l_2_total = sqrt(x_bu * x_bu
				+ z_bu * z_bu
				+ y_bu * y_bu);
			bu_l_2 += bu_l_2_total;
		}

	}


	//}
	//printf("the li_l_2 is %d \n", posi[j + 1]);
	//printf("the li_l_2 is %d \n", posi[j]);
	//printf("the li_l_2 is %f \n", li_l_2);
	//printf("the lo_l_2 is %f \n", lo_l_2);
	//printf("the cr_l_2 is %f \n", cr_l_2);
	//printf("the bu_l_2 is %f \n", bu_l_2);
	numbers[0] = li_l_2;
	numbers[1] = lo_l_2;
	numbers[2] = cr_l_2;
	numbers[3] = bu_l_2;
	return numbers;
}

void free_cal_paths_plus_result(double * numbers[], size_t size) {
	for (size_t i = 0; i < size; ++i)
		free(numbers[i]);
	
}


//double* cal_paths_plus_c(int** path_ray, int* posi, int* classes,
//  /* this is the direct translation from python to C 
// */
//	int* coord, int SizeClasses, int posiLen, int path_rayLen, double omega) {
//	/*
//	the string labels are convert into int
//	'va' ==> 0
//	'li' ==> 11
//	'lo' ==> 22
//	'cr' ==> 33
//	'bu' ==> 44
//
//	*/
//	printf("the the coord is %d %d %d \n", coord[0], coord[1], coord[2]);
//	int i, j, k;
//	//for (i = 0; i < 3; i++) {
//	//	printf("the the the last one is %d \n", path_ray[path_rayLen-1][i]);
//	//}
//	//for (i = 0; i < posiLen; i++) {
//	//	printf("the the posi is %d \n", posi[i]);
//	//}
//	//int posiL = sizeof(posi) / sizeof(posi[0]);
//	//printf("the the posi length is %d \n", posiL);
//	//int posiL = sizeof(posi) / sizeof(posi[0]);
//	//int path_rayLen = sizeof(path_ray) / sizeof(path_ray[0]);
//	//printf("the the posi length is %d \n", posiL);
//	//printf("the the posi length is %d \n", path_rayLen);
//	//printf("the omega is %f", fabs(omega));
//
//	double* numbers = malloc(4 * sizeof(double));
//	double  cr_l_2 = 0,
//		lo_l_2 = 0,
//		li_l_2 = 0,
//		bu_l_2 = 0;
//	double x_cr, y_cr, z_cr;
//	double x_li, y_li, z_li;
//	double x_lo, y_lo, z_lo;
//	double x_bu, y_bu, z_bu;
//	double cr_l_2_total, li_l_2_total, lo_l_2_total, bu_l_2_total;
//
//
//
//	if (fabs(omega) > 1 / 180.0 * PI && fabs(omega) < 179.0 / 180.0 * PI) {
//		printf("this executes in the if loop \n");
//		double total_length = sqrt((
//			path_ray[path_rayLen - 1][1] - path_ray[0][1]) * (path_ray[path_rayLen - 1][1] - path_ray[0][1]) +
//			(path_ray[path_rayLen - 1][0] - path_ray[0][0]) * (path_ray[path_rayLen - 1][0] - path_ray[0][0]) +
//			(path_ray[path_rayLen - 1][2] - path_ray[0][2]) * (path_ray[path_rayLen - 1][2] - path_ray[0][2]));
//
//		for (j = 0; j < SizeClasses; j++) {
//			if (classes[j] == 33) {
//				if (j < posiLen - 1) {
//					cr_l_2 += total_length * ((posi[j + 1] - posi[j]) / path_rayLen);
//				}
//
//				else {
//					cr_l_2 += total_length * ((path_rayLen - posi[j]) / path_rayLen);
//				}
//
//
//			}
//
//			else if (classes[j] == 11) {
//				if (j < posiLen - 1) {
//					li_l_2 += total_length * ((posi[j + 1] - posi[j]) / path_rayLen);
//				}
//
//				else {
//					li_l_2 += total_length * ((path_rayLen - posi[j]) / path_rayLen);
//				}
//			}
//
//			else if (classes[j] == 22) {
//				if (j < posiLen - 1) {
//					lo_l_2 += total_length * ((posi[j + 1] - posi[j]) / path_rayLen);
//				}
//				else {
//					lo_l_2 += total_length * ((path_rayLen - posi[j]) / path_rayLen);
//				}
//			}
//
//			else if (classes[j] == 44) {
//				if (j < posiLen - 1) {
//					bu_l_2 += total_length * ((posi[j + 1] - posi[j]) / path_rayLen);
//				}
//				else {
//					bu_l_2 += total_length * ((path_rayLen - posi[j]) / path_rayLen);
//				}
//
//			}
//
//			else {
//				printf("nothing is there");
//			}
//		}
//
//
//	}
//
//
//
//	else {
//		printf("this executes in the else loop \n");
//		for (j = 0; j < SizeClasses; j++) {
//			if (classes[j] == 33) {
//				if (j < posiLen - 1) {
//					x_cr = abs(path_ray[posi[j + 1]][2] - path_ray[posi[j]][2]);
//					z_cr = abs(path_ray[posi[j + 1]][0] - path_ray[posi[j]][0]);
//					y_cr = abs(path_ray[posi[j + 1]][1] - path_ray[posi[j]][1]);
//				}
//				else {
//					x_cr = abs(path_ray[path_rayLen-1][2] - path_ray[posi[j]][2]);
//					z_cr = abs(path_ray[path_rayLen-1][0] - path_ray[posi[j]][0]);
//					y_cr = abs(path_ray[path_rayLen-1][1] - path_ray[posi[j]][1]);
//				}
//				cr_l_2_total = sqrt((x_cr + 0.5) * (x_cr + 0.5)
//					+ (z_cr + 0.5) * (z_cr + 0.5) +
//					(y_cr + 0.5) * (y_cr + 0.5));
//				cr_l_2 += cr_l_2_total;
//			}
//			else if (classes[j] == 11) {
//				if (j < posiLen - 1) {
//					x_li = abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j]][2]);
//					z_li = abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j]][0]);
//					y_li = abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j]][1]);
//
//				}
//				else {
//					x_li = abs(path_ray[path_rayLen-1][2] - path_ray[posi[j]][2]);
//					z_li = abs(path_ray[path_rayLen-1][0] - path_ray[posi[j]][0]);
//					y_li = abs(path_ray[path_rayLen-1][1] - path_ray[posi[j]][1]);
//				}
//				li_l_2_total = sqrt(x_li * x_li
//					+ z_li * z_li
//					+ y_li * y_li);
//				li_l_2 += li_l_2_total;
//			}
//			else if (classes[j] == 22) {
//				if (j < posiLen - 1) {
//					x_lo = abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j]][2]);
//					z_lo = abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j]][0]);
//					y_lo = abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j]][1]);
//				}
//				else {
//					x_lo = abs(path_ray[path_rayLen-1][2] - path_ray[posi[j]][2]);
//					z_lo = abs(path_ray[path_rayLen-1][0] - path_ray[posi[j]][0]);
//					y_lo = abs(path_ray[path_rayLen-1][1] - path_ray[posi[j]][1]);
//
//				}
//				lo_l_2_total = sqrt(x_lo * x_lo
//					+ z_lo * z_lo
//					+ y_lo * y_lo);
//				lo_l_2 += lo_l_2_total;
//			}
//			else if (classes[j] == 44) {
//				if (j < posiLen - 1) {
//					x_bu = abs(path_ray[posi[j + 1] - 1][2] - path_ray[posi[j]][2]);
//					z_bu = abs(path_ray[posi[j + 1] - 1][0] - path_ray[posi[j]][0]);
//					y_bu = abs(path_ray[posi[j + 1] - 1][1] - path_ray[posi[j]][1]);
//
//				}
//
//				else {
//					x_bu = abs(path_ray[path_rayLen-1][2] - path_ray[posi[j]][2]);
//					z_bu = abs(path_ray[path_rayLen-1][0] - path_ray[posi[j]][0]);
//					y_bu = abs(path_ray[path_rayLen-1][1] - path_ray[posi[j]][1]);
//				}
//
//				bu_l_2_total = sqrt(x_bu * x_bu
//					+ z_bu * z_bu
//					+ y_bu * y_bu);
//				bu_l_2 += bu_l_2_total;
//			}
//
//		}
//
//
//	}
//
//	numbers[0] = li_l_2;
//	numbers[1] = lo_l_2;
//	numbers[2] = cr_l_2;
//	numbers[3] = bu_l_2;
//	return numbers;
//}