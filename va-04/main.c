#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <cstring>
#include <iostream>
#include <fstream>

using namespace std;



#define MAXCORNERS 1000


typedef struct _PARAM {
    int win_size;
    double min_dist;
    double quality;
    double umbralcc;
    int use_harris;
    int block_size;
    int roiCorr;
    int umbral;
} PARAM;

#define HAB Homografias[0]
#define HBC Homografias[1]
#define HCD Homografias[2]
#define HDE Homografias[3]
#define HEF Homografias[4]
#define HFG Homografias[5]
#define HGH Homografias[6]


#define IMAGEN_A imagenes[0]
#define IMAGEN_B imagenes[1]
#define IMAGEN_C imagenes[2]
#define IMAGEN_D imagenes[3]
#define IMAGEN_E imagenes[4]
#define IMAGEN_F imagenes[5]
#define IMAGEN_G imagenes[6]
#define IMAGEN_H imagenes[7]


typedef struct imagenesChess
{
    IplImage * imagen;
    CvPoint2D32f  corners[12*12];
    int count;
    CvSize patternSize;
    int found;
} estructuraImagenes;




float distancia(CvPoint2D32f p1, CvPoint2D32f p2)
{
    float b = sqrt((p1.x-p2.x)*(p1.x-p2.x)+ (p1.y-p2.y)*(p1.y-p2.y));
    return b;
}



CvRect calculaROI(CvPoint2D32f p1, int ancho ,int alto, CvSize size)
{
   int x=cvRound(p1.x)- ancho/2;
   int y=cvRound(p1.y)- alto/2;
   if(x<0 || y<0 ||x+ancho>size.width ||y+alto>size.height) x=-1;

   return(cvRect(x,y,ancho,alto));
}


void pintaI(char * name, IplImage * im);
void primeraParte(char * imagen);

void puntosEsquina(IplImage * im, CvPoint2D32f * corners, int *cornerCount);
void puntosHarris(IplImage * im, CvPoint2D32f * corners, int *cornerCount);

void segundaParte(char * im1, char * im2);

void terceraParte(char * im1, char * im2);

void CalcularPuntosCorrespondencias(IplImage * im1, IplImage * im2, CvMat ** src, CvMat ** dst, PARAM param, int *num);






void practica2_a()
{
    FILE * datos[5];
    IplImage * imagenes[5];
    CvMat * src = cvCreateMat(2, 256, CV_64FC1);
    CvMat * dst = cvCreateMat(2, 256, CV_64FC1);
    IplImage * imagenAuxiliar = cvCreateImage(cvSize(1000, 700), IPL_DEPTH_8U, 1);
    float aux = 0;
    CvMat * H = cvCreateMat(3, 3, CV_64FC1);

    CvMat * HTrans = cvCreateMat(3,3,CV_64FC1);
    cvSet2D(HTrans, 0, 0, cvScalar(1));
    cvSet2D(HTrans, 1, 1, cvScalar(1));
    cvSet2D(HTrans, 2, 2, cvScalar(1));
    cvSet2D(HTrans, 0, 2, cvScalar(100));
    cvSet2D(HTrans, 1, 2, cvScalar(100));

    datos[0] = fopen("imagenes/data1.txt", "r");
    datos[1] = fopen("imagenes/data2.txt", "r");
    datos[2] = fopen("imagenes/data3.txt", "r");
    datos[3] = fopen("imagenes/data4.txt", "r");
    datos[4] = fopen("imagenes/data5.txt", "r");

    imagenes[0] = cvLoadImage("imagenes/CalibIm1.PNG", 0);
    imagenes[1] = cvLoadImage("imagenes/CalibIm2.PNG", 0);
    imagenes[2] = cvLoadImage("imagenes/CalibIm3.PNG", 0);
    imagenes[3] = cvLoadImage("imagenes/CalibIm4.PNG", 0);
    imagenes[4] = cvLoadImage("imagenes/CalibIm5.PNG", 0);

    for(int i=0; i<4; i++)
    {
        // Leemos los datos de las dos imágenes

        for(int j=0; j<256; j++)
        {
            fscanf(datos[i], "%f",&aux);
            cvmSet(src, 0, j, aux);
            fscanf(datos[i], "%f",&aux);
            cvmSet(src, 1, j, aux);
            fscanf(datos[i+1], "%f",&aux);
            cvmSet(dst, 0, j, aux);
            fscanf(datos[i+1], "%f",&aux);
            cvmSet(dst, 1, j, aux);
        }

        cvFindHomography(src, dst, H);
        cvGEMM(HTrans, H, 1, NULL, 1, H);
        cvWarpPerspective(imagenes[i], imagenAuxiliar, H);

        pintaI("Original", imagenes[i]);
        pintaI("Modificada", imagenAuxiliar);

        rewind(datos[i+1]);
    }
}


void practica2_b(PARAM p)
{
    IplImage * imagenes[8];
    char nombres[8];
    CvMat * H = cvCreateMat(3, 3, CV_64FC1);
    CvMat * H1 = cvCreateMat(3, 3, CV_64FC1);
    CvMat * src = cvCreateMat(2, 256, CV_64FC1);
    CvMat * dst = cvCreateMat(2, 256, CV_64FC1);
    int num = 0;
    IplImage * imagenAuxiliar = cvCreateImage(cvSize(500, 500), IPL_DEPTH_8U, 1);

    CvMat * HTrans = cvCreateMat(3,3,CV_64FC1);
    cvSet2D(HTrans, 0, 0, cvScalar(1));
    cvSet2D(HTrans, 1, 1, cvScalar(1));
    cvSet2D(HTrans, 2, 2, cvScalar(1));
    cvSet2D(HTrans, 0, 2, cvScalar(100));
    cvSet2D(HTrans, 1, 2, cvScalar(100));

    imagenes[0] = cvLoadImage("imagenes/fig7.9a.PNG", 0);
    imagenes[1] = cvLoadImage("imagenes/fig7.9b.PNG", 0);
    imagenes[2] = cvLoadImage("imagenes/fig7.9c.PNG", 0);
    imagenes[3] = cvLoadImage("imagenes/fig7.9d.PNG", 0);
    imagenes[4] = cvLoadImage("imagenes/fig7.9e.PNG", 0);
    imagenes[5] = cvLoadImage("imagenes/fig7.9f.PNG", 0);
    imagenes[6] = cvLoadImage("imagenes/fig7.9g.PNG", 0);
    imagenes[7] = cvLoadImage("imagenes/fig7.9h.PNG", 0);


    for(int i=0; i<7; i++)
    {
        CalcularPuntosCorrespondencias(imagenes[i], imagenes[i+1], &src, &dst, p, &num);

        cvFindHomography(src, dst, H, CV_RANSAC, 4.5);
        cvFindHomography(dst, src, H1, CV_RANSAC, 4.5);

        cvGEMM(HTrans, H, 1, NULL, 1, H);
        cvGEMM(HTrans, H1, 1, NULL, 1, H1);

        cvWarpPerspective(imagenes[i], imagenAuxiliar, H);
        pintaI("Original", imagenes[i]);
        pintaI("Modificada", imagenAuxiliar);

        cvWarpPerspective(imagenes[i+1], imagenAuxiliar, H1);
        pintaI("Original", imagenes[i+1]);
        pintaI("Modificada", imagenAuxiliar);
    }

}


void CalcularHomografia(IplImage * src, IplImage * dst, PARAM p, CvMat ** H)
{
    CvMat * srcPuntos = cvCreateMat(2, 256, CV_64FC1);
    CvMat * dstPuntos = cvCreateMat(2, 256, CV_64FC1);
    int num = 0;

    CalcularPuntosCorrespondencias(src, dst, &srcPuntos, &dstPuntos, p, &num);
    cvFindHomography(srcPuntos, dstPuntos, *H, CV_RANSAC, 4.5);

}


void moverHomografia(CvMat ** H, float x, float y)
{
    cvSetReal2D(*H, 0, 2, cvGetReal2D(*H, 0, 1) + x);
    cvSetReal2D(*H, 1, 2, cvGetReal2D(*H, 0, 2) + y);

}

void practica2_c(PARAM p)
{
    IplImage * imagenes[10];
    CvMat * H = cvCreateMat(3, 3, CV_64FC1);
    CvMat * H1 = cvCreateMat(3, 3, CV_64FC1);
    CvMat * src = cvCreateMat(2, 256, CV_64FC1);
    CvMat * dst = cvCreateMat(2, 256, CV_64FC1);
    int num = 0;
    IplImage * imagenAuxiliar = cvCreateImage(cvSize(1000, 500), IPL_DEPTH_8U, 1);

    CvMat * HTrans = cvCreateMat(3,3,CV_64FC1);
    cvSet2D(HTrans, 0, 0, cvScalar(1));
    cvSet2D(HTrans, 1, 1, cvScalar(1));
    cvSet2D(HTrans, 2, 2, cvScalar(1));
    cvSet2D(HTrans, 0, 2, cvScalar(100));
    cvSet2D(HTrans, 1, 2, cvScalar(100));

    imagenes[0] = cvLoadImage("imagenes/mosaico002.jpg", 0);
    imagenes[1] = cvLoadImage("imagenes/mosaico003.jpg", 0);
    imagenes[2] = cvLoadImage("imagenes/mosaico004.jpg", 0);
    imagenes[3] = cvLoadImage("imagenes/mosaico005.jpg", 0);
    imagenes[4] = cvLoadImage("imagenes/mosaico006.jpg", 0);
    imagenes[5] = cvLoadImage("imagenes/mosaico007.jpg", 0);
    imagenes[6] = cvLoadImage("imagenes/mosaico008.jpg", 0);
    imagenes[7] = cvLoadImage("imagenes/mosaico009.jpg", 0);
    imagenes[8] = cvLoadImage("imagenes/mosaico010.jpg", 0);
    imagenes[9] = cvLoadImage("imagenes/mosaico011.jpg", 0);


    for(int i=0; i<7; i++)
    {
        CalcularPuntosCorrespondencias(imagenes[i], imagenes[i+1], &src, &dst, p, &num);

        cvFindHomography(src, dst, H, CV_RANSAC, 4.5);
        cvFindHomography(dst, src, H1, CV_RANSAC, 4.5);

        cvGEMM(HTrans, H, 1, NULL, 1, H);
        cvGEMM(HTrans, H1, 1, NULL, 1, H1);

        cvWarpPerspective(imagenes[i], imagenAuxiliar, H);
        pintaI("Original", imagenes[i]);
        pintaI("Modificada", imagenAuxiliar);

        cvWarpPerspective(imagenes[i+1], imagenAuxiliar, H1);
        pintaI("Original", imagenes[i+1]);
        pintaI("Modificada", imagenAuxiliar);
    }
}


void practica2_d(PARAM p)
{
    IplImage * imagenes[8];
    char nombres[8];
    CvMat * H = cvCreateMat(3, 3, CV_64FC1);
    CvMat * H1 = cvCreateMat(3, 3, CV_64FC1);
    CvMat * src = cvCreateMat(2, 256, CV_64FC1);
    CvMat * dst = cvCreateMat(2, 256, CV_64FC1);
    CvMat ** aux;
    int num = 0;
    IplImage * imagenAuxiliar = cvCreateImage(cvSize(500, 500), IPL_DEPTH_8U, 1);
    IplImage * imagenFinal = cvCreateImage(cvSize(1500, 500), IPL_DEPTH_8U, 1);

    CvMat * Homografias[8];


    for(int i=0; i<7; i++)
        Homografias[i] = cvCreateMat(3, 3, CV_64FC1);


    CvMat * HRes = cvCreateMat(3, 3, CV_64FC1);

//#define aux AB;

    CvMat * HTrans = cvCreateMat(3,3,CV_64FC1);
    cvSet2D(HTrans, 0, 0, cvScalar(1));
    cvSet2D(HTrans, 1, 1, cvScalar(1));
    cvSet2D(HTrans, 2, 2, cvScalar(1));
    cvSet2D(HTrans, 0, 2, cvScalar(500));
    cvSet2D(HTrans, 1, 2, cvScalar(200));

    imagenes[0] = cvLoadImage("imagenes/fig7.9a.PNG", 0);
    imagenes[1] = cvLoadImage("imagenes/fig7.9b.PNG", 0);
    imagenes[2] = cvLoadImage("imagenes/fig7.9c.PNG", 0);
    imagenes[3] = cvLoadImage("imagenes/fig7.9d.PNG", 0);
    imagenes[4] = cvLoadImage("imagenes/fig7.9e.PNG", 0);
    imagenes[5] = cvLoadImage("imagenes/fig7.9f.PNG", 0);
    imagenes[6] = cvLoadImage("imagenes/fig7.9g.PNG", 0);
    imagenes[7] = cvLoadImage("imagenes/fig7.9h.PNG", 0);


    CvMat * Hab,  * Hbc,  *Hcd,   *Hfe,  *Hgf,  *Hhg;
    CvMat * Homografia = cvCreateMat(3,3,CV_64FC1);
    CvMat * Haux = cvCreateMat(3, 3, CV_64FC1);

    cvWarpPerspective(imagenes[3], imagenFinal, HTrans);

    CalcularHomografia(IMAGEN_A, IMAGEN_B, p, & HAB);
    CalcularHomografia(IMAGEN_B, IMAGEN_C, p, & HBC);
    CalcularHomografia(IMAGEN_C, IMAGEN_D, p, & HCD);
    CalcularHomografia(IMAGEN_D, IMAGEN_E, p, & HDE);
    CalcularHomografia(IMAGEN_E, IMAGEN_F, p, & HEF);
    CalcularHomografia(IMAGEN_F, IMAGEN_G, p, & HFG);
    CalcularHomografia(IMAGEN_G, IMAGEN_H, p, & HGH);

    cvInvert(HDE, HDE);
    cvInvert(HEF, HEF);
    cvInvert(HFG, HFG);
    cvInvert(HGH, HGH);



    //CalcularHomografia(imagenes[4],imagenes[3], p, &Homografia);
    cvGEMM(HTrans, HDE, 1, NULL, 1, Homografia);
    cvWarpPerspective(imagenes[4], imagenFinal, Homografia, CV_INTER_LINEAR);

    //CalcularHomografia(imagenes[5], imagenes[4], p, &Haux);
    cvGEMM(Homografia, HEF, 1, NULL, 1, Homografia);
    cvWarpPerspective(imagenes[5], imagenFinal, Homografia, CV_INTER_LINEAR);


    //CalcularHomografia(imagenes[6], imagenes[5], p, &Haux);
    cvGEMM(Homografia, HFG, 1, NULL, 1, Homografia);
    cvWarpPerspective(imagenes[6], imagenFinal, Homografia, CV_INTER_LINEAR);

    //CalcularHomografia(imagenes[7], imagenes[6], p, &Haux);
    cvGEMM(Homografia, HGH, 1, NULL, 1, Homografia);
    cvWarpPerspective(imagenes[7], imagenFinal, Homografia, CV_INTER_LINEAR);





    //CalcularHomografia(imagenes[2], imagenes[3], p, &Homografia);
    cvGEMM(HTrans, HCD, 1, NULL, 1, Homografia);
    cvWarpPerspective(IMAGEN_C, imagenFinal, Homografia, CV_INTER_LINEAR);


    //CalcularHomografia(imagenes[1], imagenes[2], p, &Haux);
    cvGEMM(Homografia, HBC, 1, NULL, 1, Homografia);
    cvWarpPerspective(IMAGEN_B, imagenFinal, Homografia, CV_INTER_LINEAR);


    //CalcularHomografia(imagenes[0], imagenes[1], p, &Haux);
    cvGEMM(Homografia, HAB, 1, NULL, 1, Homografia);
    cvWarpPerspective(IMAGEN_A, imagenFinal, Homografia, CV_INTER_LINEAR);


    pintaI("Mosaico",imagenFinal);
}


void practica3_A(estructuraImagenes ** v, int * n)
{
    estructuraImagenes vector[26];
    estructuraImagenes validas[26];
    int numValidas = 0;

    char tituloCompleto[100];
    tituloCompleto[0] = '\0';
    char numero[5];
    char aux[100];
    aux[0] = '\0';

    IplImage * dst;


    for(int i=1; i<=25; i++)
    {
        sprintf(numero, "%d", i);
        strcat(tituloCompleto, "imagenes/Image");
        strcat(tituloCompleto, numero);
        strcat(tituloCompleto, ".tif");

        vector[i].imagen = cvLoadImage(tituloCompleto, 0);
        dst = cvCloneImage(vector[i].imagen);

        vector[i].patternSize = cvSize(12,12);
        vector[i].found = cvFindChessboardCorners(vector[i].imagen, vector[i].patternSize, vector[i].corners, &(vector[i].count), CV_CALIB_CB_ADAPTIVE_THRESH + CV_CALIB_CB_NORMALIZE_IMAGE);
        cvFindCornerSubPix(vector[i].imagen, vector[i].corners, vector[i].count, cvSize(3,3), cvSize(-1,-1), cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03));
        cvDrawChessboardCorners(dst, vector[i].patternSize, vector[i].corners, vector[i].count, vector[i].found);

        if(vector[i].found != 0)
        {
            validas[numValidas] = vector[i];
            numValidas++;
        }

        tituloCompleto[0] = '\0';
        strcat(aux, "Imagen");
        strcat(aux, numero);

        if(vector[i].found != 0)
            pintaI(aux , dst);

        aux[0] = '\0';
    }

    *v = new estructuraImagenes [numValidas];

    for(int i=0; i<numValidas; i++)
        (*v)[i] = validas[i];

    *n = numValidas;
}



/**
    Función que realiza el apartado C de la práctica 3
    v representa las imágenes sobre las que queremos actuar
    n es el número de imágenes que contiene v
    cameraMatrix es un parámetro de salida, que nos devuelve la matriz de calibración necesaria para el apartado C
    distCoeffs es otro parámetro de salida, que nos devuelve los coeficientes de distorsión para usar en el apartado C
*/

void practica3_B(estructuraImagenes * validas, int numValidas, CvMat ** camMatrix, CvMat ** dCoeffs)
{
    CvSize size = cvSize(validas[0].imagen->width, validas[0].imagen->height);

    CvPoint3D32f *  objectPoints = new CvPoint3D32f[144 * numValidas];   // Coordenadas del mundo
    CvPoint2D32f * imagePoints = new CvPoint2D32f[144 * numValidas];    // Coordenadas de la imagen
    int * npoints = new int[numValidas];
    CvMat * proyectados = cvCreateMat(2, 144, CV_32FC1);


    // Almacenamos las posiciones de los puntos en el mundo
    for(int i=0; i<numValidas; i++)
    {
        for(int j=0; j<12; j++)
        {
            for(int k=0; k<12; k++)
            {
                objectPoints[(i*144) + j*12 + k].x = j;
                objectPoints[(i*144) + j*12 + k].y = k;
                objectPoints[(i*144) + j*12 + k].z = 0;
            }
        }
    }


    // Almacenamos las posiciones de los puntos en la imagen

    for(int i=0; i<numValidas; i++)
    {
        for(int j=0; j<validas[i].count; j++)
            imagePoints[i*144 + j] = validas[i].corners[j];
    }

    // Rellenamos el vector de npoints que representa cuantos puntos tenemos por imagen

    for(int i=0; i<numValidas; i++)
        npoints[i] = 144;

    CvMat * cameraMatrix = cvCreateMat(3, 3, CV_32FC1);
    CvMat * rvec, * tvec;

    CvMat * rvecs = cvCreateMat(numValidas, 3, CV_32FC1);
    CvMat * tvecs = cvCreateMat(numValidas, 3, CV_32FC1);
    CvMat * distortionCoeffs = cvCreateMat(1, 4, CV_32F);

    rvec = cvCreateMat(1, 3, CV_32FC1);
    tvec = cvCreateMat(1, 3, CV_32FC1);


    CvMat * rvecsP[12];
    CvMat * tvecsP[12];

    for(int i=0; i<12; i++)
    {
        rvecsP[i] = cvCreateMat(1, 3, CV_32FC1);
        tvecsP[i] = cvCreateMat(1, 3, CV_32FC1);
    }


    // Llamamos a la función para conseguir los parámetros intrínsecos de la cámara
    cvInitIntrinsicParams2D(
        &cvMat(1, 144*numValidas, CV_32FC3, objectPoints),
        &cvMat(1, 144*numValidas, CV_32FC2, imagePoints),
        &cvMat(1, numValidas, CV_32SC1, npoints),
        size,
        cameraMatrix);


    float error = 0;
    float auxiliar = 0;



    // Obtenemos ahora los parámetros extrínsecos

    for(int i=0; i<numValidas; i++)
    {
        cvFindExtrinsicCameraParams2(
            &cvMat(1, 144, CV_32FC3, objectPoints),
            &cvMat(1, 144, CV_32FC2, imagePoints + i*144),
            cameraMatrix,
            NULL,
            rvecsP[i],
            tvecsP[i]);

            cvProjectPoints2(
                &cvMat(1, 144, CV_32FC3, objectPoints),
                rvecsP[i],
                tvecsP[i],
                cameraMatrix,
                NULL,
                proyectados);

            float x1, x2, y1, y2;

            for(int j=0; j<144; j++)
            {
                x1 = validas[i].corners[j].x;
                x2 = cvGetReal1D(proyectados, j * 2);

                y1 = validas[i].corners[j].y;
                y2 = cvGetReal1D(proyectados, j);
                auxiliar += sqrt(pow(validas[i].corners[j].x - cvGetReal2D(proyectados, 0, j), 2) + pow(validas[i].corners[j].y - cvGetReal2D(proyectados, 1, j),2));
            }

            auxiliar /= 144;
            error += auxiliar;
            auxiliar = 0;
    }

    error /= numValidas;


    cout << "El error que obtenemos sin tener en cuenta los coeficientes de distorsión es: " << error << endl << endl;



    error = 0;
    auxiliar = 0;
    // Ahora llamamos a la función de calibración de la cámara con los resultados obtenidos por las anteriores funciones

    cvCalibrateCamera2(
        &cvMat(1, 144*numValidas, CV_32FC3, objectPoints),
        &cvMat(1, 144*numValidas, CV_32FC2, imagePoints),
        &cvMat(1, numValidas, CV_32SC1, npoints),
        size,
        cameraMatrix,
        distortionCoeffs,
        rvecs,
        tvecs);



    for(int i=0; i<numValidas; i++)
    {

        cvSet2D(rvec, 0, 0, cvScalar(cvGetReal2D(rvecs, i, 0)));
        cvSet2D(rvec, 0, 1, cvScalar(cvGetReal2D(rvecs, i, 1)));
        cvSet2D(rvec, 0, 2, cvScalar(cvGetReal2D(rvecs, i, 2)));

        cvSet2D(tvec, 0, 0, cvScalar(cvGetReal2D(tvecs, i, 0)));
        cvSet2D(tvec, 0, 1, cvScalar(cvGetReal2D(tvecs, i, 1)));
        cvSet2D(tvec, 0, 2, cvScalar(cvGetReal2D(tvecs, i, 2)));

        cvProjectPoints2(
        &cvMat(1, 144, CV_32FC3, objectPoints),
        rvec,
        tvec,
        cameraMatrix,
        distortionCoeffs,
        proyectados);


        for(int j=0; j<144; j++)
        {
            auxiliar += sqrt(pow(validas[i].corners[j].x - cvGetReal2D(proyectados, 0, j), 2) + pow(validas[i].corners[j].y - cvGetReal2D(proyectados, 1, j),2));
        }

        auxiliar /= 144;
        error += auxiliar;
        auxiliar = 0;
    }

    error /= numValidas;

    cout << "El error que obtenemos teniendo en cuenta los coeficientes de distorsión es: " << error << endl << endl;


    *camMatrix = cameraMatrix;
    *dCoeffs = distortionCoeffs;
}


void practica3_C(estructuraImagenes * validas, int numValidas, CvMat * camMatrix, CvMat * dCoeffs)
{
    IplImage * dst = cvCreateImage(cvSize(validas[0].imagen->width, validas[0].imagen->height), validas[0].imagen->depth, validas[0].imagen->nChannels);

    for(int i=0; i<numValidas; i++)
    {
        cvUndistort2(
            validas[i].imagen,
            dst,
            camMatrix,
            dCoeffs);

        cvNamedWindow("Imagen Original");
        cvNamedWindow("Imagen Rectificada");
        cvShowImage("Imagen Original", validas[i].imagen);
        cvShowImage("Imagen Rectificada", dst);
        cvWaitKey();
        cvDestroyWindow("Imagen Original");
        cvDestroyWindow("Imagen Rectificada");
    }


}




float obtenerPuntoLinea(CvMat * line, int x, int i)
{
    float a = cvGetReal2D(line, 0, i);
    float b = cvGetReal2D(line, 1, i);
    float c = cvGetReal2D(line, 2, i);

    return (((-a * x) -c)/b);
}


void calcularPuntoCorte(CvMat * lines, int i1, int i2, float * x, float * y)
{
    float a = cvGetReal2D(lines, 0, i1);
    float b = cvGetReal2D(lines, 1, i1);
    float c = cvGetReal2D(lines, 2, i1);

    float d = cvGetReal2D(lines, 0, i2);
    float e = cvGetReal2D(lines, 1, i2);
    float f = cvGetReal2D(lines, 2, i2);

    *y = ((-a*f) + c*d)/((-d*b) + a*e);
    *x = ((-c) - b*(*y))/a;

}



float pendientePerpendicular(CvMat * line, int i)
{
    float b = cvGetReal2D(line, 1, i);
    float a = cvGetReal2D(line, 0, i);

    return (b/a);
}


CvMat * rectaPerpendicular(CvMat * line, int i, float x, float y)
{
    float a, b, c, m;
    CvMat * recta = cvCreateMat(3, 1, CV_32FC1);

    m = pendientePerpendicular(line, i);

    a = 1;
    b = -m;
    c = m*x - y;

    cvSetReal2D(recta, 0, 0, a);
    cvSetReal2D(recta, 1, 0, b);
    cvSetReal2D(recta, 2, 0, c);

    return recta;
}


float distanciaPuntoRecta(CvMat * lines, int i, int x, int y)
{
    float a = cvGetReal2D(lines, 0, i);
    float b = cvGetReal2D(lines, 1, i);
    float c = cvGetReal2D(lines, 2, i);

    return(abs(a*x + b*y +c)/sqrt(pow(a,2) + pow(b,2)));
}



void mostrarMatriz(CvMat * m)
{
    for(int i=0; i<m->rows; i++)
    {
        for(int j=0; j<m->cols; j++)
        {
            cout << cvGetReal2D(m, i, j) << "  ";
        }
        cout << endl;
    }
}


void practica4_A()
{
    CvMat * points1 = cvCreateMat(2, 8, CV_32FC1);
    CvMat * points2 = cvCreateMat(2, 8, CV_32FC1);
    CvMat * F = cvCreateMat(3, 3, CV_32FC1);

    CvMat * lines1 = cvCreateMat(3, 8, CV_32FC1);
    CvMat * lines2 = cvCreateMat(3, 8, CV_32FC1);

    IplImage * img1 = cvLoadImage("./imagenes/basement00.tif", 0);
    IplImage * img2 = cvLoadImage("./imagenes/basement01.tif", 0);

    CvPoint ptsLineas1[8][2];
    CvPoint ptsLineas2[8][2];




    // Introducimos los puntos en correspondencias que hemos recogido a mano
    cvSetReal2D(points1, 0, 0, 197);
    cvSetReal2D(points1, 1, 0, 20);
    cvSetReal2D(points2, 0, 0, 194);
    cvSetReal2D(points2, 1, 0, 9);


    cvSetReal2D(points1, 0, 1, 106);
    cvSetReal2D(points1, 1, 1, 460);
    cvSetReal2D(points2, 0, 1, 95);
    cvSetReal2D(points2, 1, 1, 482);


    cvSetReal2D(points1, 0, 2, 468);
    cvSetReal2D(points1, 1, 2, 399);
    cvSetReal2D(points2, 0, 2, 493);
    cvSetReal2D(points2, 1, 2, 427);


    cvSetReal2D(points1, 0, 3, 369);
    cvSetReal2D(points1, 1, 3, 69);
    cvSetReal2D(points2, 0, 3, 378);
    cvSetReal2D(points2, 1, 3, 62);


    cvSetReal2D(points1, 0, 4, 229);
    cvSetReal2D(points1, 1, 4, 435);
    cvSetReal2D(points2, 0, 4, 227);
    cvSetReal2D(points2, 1, 4, 454);


    cvSetReal2D(points1, 0, 5, 299);
    cvSetReal2D(points1, 1, 5, 184);
    cvSetReal2D(points2, 0, 5, 301);
    cvSetReal2D(points2, 1, 5, 185);


    cvSetReal2D(points1, 0, 6, 184);
    cvSetReal2D(points1, 1, 6, 198);
    cvSetReal2D(points2, 0, 6, 182);
    cvSetReal2D(points2, 1, 6, 199);


    cvSetReal2D(points1, 0, 7, 401);
    cvSetReal2D(points1, 1, 7, 464);
    cvSetReal2D(points2, 0, 7, 413);
    cvSetReal2D(points2, 1, 7, 489);

    CvPoint p;

    for(int i=0; i<8; i++)
    {
        p.x = cvGetReal2D(points1, 0, i);
        p.y = cvGetReal2D(points1, 1, i);

        cvCircle(img1, p, 3, CvScalar());

        p.x = cvGetReal2D(points2, 0, i);
        p.y = cvGetReal2D(points2, 1, i);

        cvCircle(img2, p, 3, CvScalar());
    }



    // Hallamos la matriz fundamental
    cvFindFundamentalMat(points1, points2, F, CV_FM_8POINT);

    cout << endl <<"La Matriz fundamental (F) hallada es: " << endl;
    mostrarMatriz(F);

    // Hallamos las lineas epipolares
    cvComputeCorrespondEpilines(points2, 2, F, lines1);
    cvComputeCorrespondEpilines(points1, 1, F, lines2);


    float ep1x, ep1y, ep2x, ep2y;

    calcularPuntoCorte(lines1, 2, 3, &ep1x, &ep1y);
    calcularPuntoCorte(lines2, 0, 1, &ep2x, &ep2y);

    cout << endl << "El epipolo izquierdo es (" << ep1x << ", " << ep1y << ")." << endl;
    cout << endl << "El epipolo derecho es (" << ep2x << ", " << ep2y << ")." << endl;

    // Ahora tenemos que pintar las lineas, para lo que antes hallaremos dos puntos de cada linea

    for(int i=0; i<8; i++)
    {
        ptsLineas1[i][0].x = 0;
        ptsLineas1[i][0].y = obtenerPuntoLinea(lines1, 0, i);
        ptsLineas1[i][1].x = img1->width;
        ptsLineas1[i][1].y = obtenerPuntoLinea(lines1, img1->width, i);


        ptsLineas2[i][0].x = 0;
        ptsLineas2[i][0].y = obtenerPuntoLinea(lines2, 0, i);
        ptsLineas2[i][1].x = img2->width;
        ptsLineas2[i][1].y = obtenerPuntoLinea(lines2, img2->width, i);
    }

    // Pintamos las líneas en las imágenes

    for(int i=0; i<8; i++)
    {
        cvLine(img1, ptsLineas1[i][0], ptsLineas1[i][1], CvScalar());
        cvLine(img2, ptsLineas2[i][0], ptsLineas2[i][1], CvScalar());
    }


    cout << endl;
    cout << "Mostramos las líneas epipolares en cada imagen, junto con los puntos en correspondencia usados. " << endl;
    cout << "(para pasar de imagen pulsar una tecla)" << endl;
    pintaI("Primera", img1);
    pintaI("Segunda", img2);


    // Ahora debemos calcular la matriz de proyección asociada a cada cámara

    CvMat * Pi = cvCreateMat(3, 4, CV_32FC1);

    cvSetReal2D(Pi, 0, 0, 1);
    cvSetReal2D(Pi, 0, 1, 0);
    cvSetReal2D(Pi, 0, 2, 0);
    cvSetReal2D(Pi, 0, 3, 0);

    cvSetReal2D(Pi, 1, 0, 0);
    cvSetReal2D(Pi, 1, 1, 1);
    cvSetReal2D(Pi, 1, 2, 0);
    cvSetReal2D(Pi, 1, 3, 0);

    cvSetReal2D(Pi, 2, 0, 0);
    cvSetReal2D(Pi, 2, 1, 0);
    cvSetReal2D(Pi, 2, 2, 1);
    cvSetReal2D(Pi, 2, 3, 0);


    CvMat * Pd = cvCreateMat(3, 4, CV_32FC1);

    CvMat * ex = cvCreateMat(3, 3, CV_32FC1);

    cvSetReal2D(ex, 0, 0, 0);
    cvSetReal2D(ex, 0, 1, -ep1x);
    cvSetReal2D(ex, 0, 2, ep1y);

    cvSetReal2D(ex, 0, 0, ep1x);
    cvSetReal2D(ex, 0, 1, 0);
    cvSetReal2D(ex, 0, 2, -ep1x);

    cvSetReal2D(ex, 0, 0, -ep1y);
    cvSetReal2D(ex, 0, 1, ep1x);
    cvSetReal2D(ex, 0, 2, 0);

    CvMat * resultado = cvCreateMat(3, 3, CV_32FC1);

    cvGEMM(ex, F, 1, NULL, 1, resultado);


    for(int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
        {
            cvSetReal2D(Pd, i, j, cvGetReal2D(resultado, i, j));
        }
    }

    cvSetReal2D(Pd, 0, 3, ep2x);
    cvSetReal2D(Pd, 1, 3, ep2y);
    cvSetReal2D(Pd, 2, 3, 0);

    cout << endl;
    cout << "Las matrices de proyeccion halladas son las siguientes: " << endl << endl;
    mostrarMatriz(Pi);
    cout << endl << endl;
    mostrarMatriz(Pd);

    // Ya hemos encontrado las matrices de proyección, ahora calculamos la distancia ortogonal media

    float media = 0;

    for(int i=0; i<8; i++)
    {
        media += distanciaPuntoRecta(lines1, i, cvGetReal2D(points1, 0, i), cvGetReal2D(points1, 1, i));
        media += distanciaPuntoRecta(lines2, i, cvGetReal2D(points2, 0, i), cvGetReal2D(points2, 1, i));
    }

    media /= 16;

    cout << endl << "El error medio es: " << media << endl;
}


void Practica4_A_Vmort()
{
    CvMat * points1 = cvCreateMat(2, 640, CV_32FC1);
    CvMat * points2 = cvCreateMat(2, 640, CV_32FC1);

    CvMat * F = cvCreateMat(3, 3, CV_32FC1);

    CvMat * lines1 = cvCreateMat(3, 640, CV_32FC1);
    CvMat * lines2 = cvCreateMat(3, 640, CV_32FC1);

    IplImage * img1 = cvLoadImage("./imagenes/Vmort1.pgm", 0);
    IplImage * img2 = cvLoadImage("./imagenes/Vmort2.pgm", 0);

    CvPoint ptsLineas1[640][2];
    CvPoint ptsLineas2[640][2];

    fstream file1("./imagenes/Vmort1.LMSMatch", fstream::in);
    fstream file2("./imagenes/Vmort2.LMSMatch", fstream::in);

    // Leemos los datos de los archivos

    float aux;

    for(int i=0; i<640; i++)
    {
        file1 >> aux;
        cvSetReal2D(points1, 0, i, aux);
        file1 >> aux;
        cvSetReal2D(points1, 1, i, aux);

        file2 >> aux;
        cvSetReal2D(points2, 0, i, aux);
        file2 >> aux;
        cvSetReal2D(points2, 1, i, aux);
    }
    CvPoint p;

    for(int i=0; i<640; i++)
    {
        p.x = cvGetReal2D(points1, 0, i);
        p.y = cvGetReal2D(points1, 1, i);

        cvCircle(img1, p, 3, CvScalar());

        p.x = cvGetReal2D(points2, 0, i);
        p.y = cvGetReal2D(points2, 1, i);

        cvCircle(img2, p, 3, CvScalar());
    }

    // Hallamos la matriz fundamental
    cvFindFundamentalMat(points1, points2, F, CV_FM_8POINT);

    cout << endl <<"La Matriz fundamental (F) hallada es: " << endl;
    mostrarMatriz(F);


    // Hallamos las lineas epipolares
    cvComputeCorrespondEpilines(points2, 2, F, lines1);
    cvComputeCorrespondEpilines(points1, 1, F, lines2);



    float ep1x, ep1y, ep2x, ep2y;

    calcularPuntoCorte(lines1, 2, 3, &ep1x, &ep1y);
    calcularPuntoCorte(lines2, 0, 1, &ep2x, &ep2y);

    cout << endl << "El epipolo izquierdo es (" << ep1x << ", " << ep1y << ")." << endl;
    cout << endl << "El epipolo derecho es (" << ep2x << ", " << ep2y << ")." << endl;

    // Ahora tenemos que pintar las lineas, para lo que antes hallaremos dos puntos de cada linea

    for(int i=0; i<640; i++)
    {
        ptsLineas1[i][0].x = 0;
        ptsLineas1[i][0].y = obtenerPuntoLinea(lines1, 0, i);
        ptsLineas1[i][1].x = img1->width;
        ptsLineas1[i][1].y = obtenerPuntoLinea(lines1, img1->width, i);


        ptsLineas2[i][0].x = 0;
        ptsLineas2[i][0].y = obtenerPuntoLinea(lines2, 0, i);
        ptsLineas2[i][1].x = img2->width;
        ptsLineas2[i][1].y = obtenerPuntoLinea(lines2, img2->width, i);
    }

    // Pintamos las líneas en las imágenes

    for(int i=0; i<640; i++)
    {
        cvLine(img1, ptsLineas1[i][0], ptsLineas1[i][1], CvScalar());
        cvLine(img2, ptsLineas2[i][0], ptsLineas2[i][1], CvScalar());
    }


    cout << endl;
    cout << "Mostramos las líneas epipolares en cada imagen, junto con los puntos en correspondencia usados. " << endl;
    cout << "(para pasar de imagen pulsar una tecla)" << endl;

    pintaI("Primera", img1);
    pintaI("Segunda", img2);



        // Ahora debemos calcular la matriz de proyección asociada a cada cámara

    CvMat * Pi = cvCreateMat(3, 4, CV_32FC1);

    cvSetReal2D(Pi, 0, 0, 1);
    cvSetReal2D(Pi, 0, 1, 0);
    cvSetReal2D(Pi, 0, 2, 0);
    cvSetReal2D(Pi, 0, 3, 0);

    cvSetReal2D(Pi, 1, 0, 0);
    cvSetReal2D(Pi, 1, 1, 1);
    cvSetReal2D(Pi, 1, 2, 0);
    cvSetReal2D(Pi, 1, 3, 0);

    cvSetReal2D(Pi, 2, 0, 0);
    cvSetReal2D(Pi, 2, 1, 0);
    cvSetReal2D(Pi, 2, 2, 1);
    cvSetReal2D(Pi, 2, 3, 0);


    CvMat * Pd = cvCreateMat(3, 4, CV_32FC1);

    CvMat * ex = cvCreateMat(3, 3, CV_32FC1);

    cvSetReal2D(ex, 0, 0, 0);
    cvSetReal2D(ex, 0, 1, -ep1x);
    cvSetReal2D(ex, 0, 2, ep1y);

    cvSetReal2D(ex, 0, 0, ep1x);
    cvSetReal2D(ex, 0, 1, 0);
    cvSetReal2D(ex, 0, 2, -ep1x);

    cvSetReal2D(ex, 0, 0, -ep1y);
    cvSetReal2D(ex, 0, 1, ep1x);
    cvSetReal2D(ex, 0, 2, 0);

    CvMat * resultado = cvCreateMat(3, 3, CV_32FC1);

    cvGEMM(ex, F, 1, NULL, 1, resultado);


    for(int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
        {
            cvSetReal2D(Pd, i, j, cvGetReal2D(resultado, i, j));
        }
    }

    cvSetReal2D(Pd, 0, 3, ep2x);
    cvSetReal2D(Pd, 1, 3, ep2y);
    cvSetReal2D(Pd, 2, 3, 0);

    cout << endl;
    cout << "Las matrices de proyeccion halladas son las siguientes: " << endl << endl;
    mostrarMatriz(Pi);
    cout << endl << endl;
    mostrarMatriz(Pd);


    // Ya hemos encontrado las matrices de proyección, ahora calculamos la distancia ortogonal media

    float media = 0;

    for(int i=0; i<640; i++)
    {
        media += distanciaPuntoRecta(lines1, i, cvGetReal2D(points1, 0, i), cvGetReal2D(points1, 1, i));
        media += distanciaPuntoRecta(lines2, i, cvGetReal2D(points2, 0, i), cvGetReal2D(points2, 1, i));
    }

    media /= 1200;

    cout << endl << "El error medio es: " << media << endl;


}



void Practica4_B()
{
    CvMat * points1 = cvCreateMat(2, 1000, CV_32FC1);
    CvMat * points2 = cvCreateMat(2, 1000, CV_32FC1);
    CvMat * F = cvCreateMat(3, 3, CV_32FC1);
    int numPuntos = 1000;



    IplImage * img1 = cvLoadImage("./imagenes/basement00.tif", 0);
    IplImage * img2 = cvLoadImage("./imagenes/basement01.tif", 0);

    CvPoint ptsLineas1[1000][2];
    CvPoint ptsLineas2[1000][2];

    PARAM param;
    param.win_size = 11;
    param.block_size = 5;
    param.min_dist = 2;
    param.quality = 0.001;
    param.roiCorr = 7;
    param.umbral = 100;
    param.umbralcc = 0.75;
    param.use_harris = 1;

    cout << endl << "Calculando puntos en correspondencias..." << endl;

    CalcularPuntosCorrespondencias(img1, img2, &points1, &points2, param, &numPuntos);

    CvMat * lines1 = cvCreateMat(3, numPuntos, CV_32FC1);
    CvMat * lines2 = cvCreateMat(3, numPuntos, CV_32FC1);


    CvPoint p;

    for(int i=0; i<numPuntos; i++)
    {
        p.x = cvGetReal2D(points1, 0, i);
        p.y = cvGetReal2D(points1, 1, i);

        cvCircle(img1, p, 3, CvScalar());

        p.x = cvGetReal2D(points2, 0, i);
        p.y = cvGetReal2D(points2, 1, i);

        cvCircle(img2, p, 3, CvScalar());
    }



    // Hallamos la matriz fundamental
    cvFindFundamentalMat(points1, points2, F, CV_FM_RANSAC, 2);

    cout << endl <<"La Matriz fundamental (F) hallada es: " << endl;
    mostrarMatriz(F);

    // Hallamos las lineas epipolares
    cvComputeCorrespondEpilines(points2, 2, F, lines1);
    cvComputeCorrespondEpilines(points1, 1, F, lines2);


    float ep1x, ep1y, ep2x, ep2y;

    calcularPuntoCorte(lines1, 0, 1, &ep1x, &ep1y);
    calcularPuntoCorte(lines2, 0, 1, &ep2x, &ep2y);

    cout << endl << "El epipolo izquierdo es (" << ep1x << ", " << ep1y << ")." << endl;
    cout << endl << "El epipolo derecho es (" << ep2x << ", " << ep2y << ")." << endl;

    // Ahora tenemos que pintar las lineas, para lo que antes hallaremos dos puntos de cada linea

    for(int i=0; i<numPuntos; i++)
    {
        ptsLineas1[i][0].x = 0;
        ptsLineas1[i][0].y = obtenerPuntoLinea(lines1, 0, i);
        ptsLineas1[i][1].x = img1->width;
        ptsLineas1[i][1].y = obtenerPuntoLinea(lines1, img1->width, i);


        ptsLineas2[i][0].x = 0;
        ptsLineas2[i][0].y = obtenerPuntoLinea(lines2, 0, i);
        ptsLineas2[i][1].x = img2->width;
        ptsLineas2[i][1].y = obtenerPuntoLinea(lines2, img2->width, i);
    }

    // Pintamos las líneas en las imágenes

    for(int i=0; i<numPuntos; i++)
    {
        cvLine(img1, ptsLineas1[i][0], ptsLineas1[i][1], CvScalar());
        cvLine(img2, ptsLineas2[i][0], ptsLineas2[i][1], CvScalar());
    }


    cout << endl;
    cout << "Mostramos las líneas epipolares en cada imagen, junto con los puntos en correspondencia usados. " << endl;
    cout << "(para pasar de imagen pulsar una tecla)" << endl;

    pintaI("Primera", img1);
    pintaI("Segunda", img2);

    // Ahora debemos calcular la matriz de proyección asociada a cada cámara

    CvMat * Pi = cvCreateMat(3, 4, CV_32FC1);

    cvSetReal2D(Pi, 0, 0, 1);
    cvSetReal2D(Pi, 0, 1, 0);
    cvSetReal2D(Pi, 0, 2, 0);
    cvSetReal2D(Pi, 0, 3, 0);

    cvSetReal2D(Pi, 1, 0, 0);
    cvSetReal2D(Pi, 1, 1, 1);
    cvSetReal2D(Pi, 1, 2, 0);
    cvSetReal2D(Pi, 1, 3, 0);

    cvSetReal2D(Pi, 2, 0, 0);
    cvSetReal2D(Pi, 2, 1, 0);
    cvSetReal2D(Pi, 2, 2, 1);
    cvSetReal2D(Pi, 2, 3, 0);


    CvMat * Pd = cvCreateMat(3, 4, CV_32FC1);

    CvMat * ex = cvCreateMat(3, 3, CV_32FC1);

    cvSetReal2D(ex, 0, 0, 0);
    cvSetReal2D(ex, 0, 1, -ep1x);
    cvSetReal2D(ex, 0, 2, ep1y);

    cvSetReal2D(ex, 0, 0, ep1x);
    cvSetReal2D(ex, 0, 1, 0);
    cvSetReal2D(ex, 0, 2, -ep1x);

    cvSetReal2D(ex, 0, 0, -ep1y);
    cvSetReal2D(ex, 0, 1, ep1x);
    cvSetReal2D(ex, 0, 2, 0);

    CvMat * resultado = cvCreateMat(3, 3, CV_32FC1);

    cvGEMM(ex, F, 1, NULL, 1, resultado);


    for(int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
        {
            cvSetReal2D(Pd, i, j, cvGetReal2D(resultado, i, j));
        }
    }

    cvSetReal2D(Pd, 0, 3, ep2x);
    cvSetReal2D(Pd, 1, 3, ep2y);
    cvSetReal2D(Pd, 2, 3, 0);

    cout << endl;
    cout << "Las matrices de proyeccion halladas son las siguientes: " << endl << endl;
    mostrarMatriz(Pi);
    cout << endl << endl;
    mostrarMatriz(Pd);

    // Ya hemos encontrado las matrices de proyección, ahora calculamos la distancia ortogonal media

    float media = 0;

    for(int i=0; i<numPuntos; i++)
    {
        media += distanciaPuntoRecta(lines1, i, cvGetReal2D(points1, 0, i), cvGetReal2D(points1, 1, i));
        media += distanciaPuntoRecta(lines2, i, cvGetReal2D(points2, 0, i), cvGetReal2D(points2, 1, i));
    }

    media /= (numPuntos*2);

    cout << endl << "El error medio es: " << media << endl;
}


void Practica4_B_Vmort()
{
    CvMat * points1 = cvCreateMat(2, 1000, CV_32FC1);
    CvMat * points2 = cvCreateMat(2, 1000, CV_32FC1);
    CvMat * F = cvCreateMat(3, 3, CV_32FC1);
    int numPuntos = 1000;



    IplImage * img1 = cvLoadImage("./imagenes/Vmort1.pgm", 0);
    IplImage * img2 = cvLoadImage("./imagenes/Vmort2.pgm", 0);
    CvPoint ptsLineas1[1000][2];
    CvPoint ptsLineas2[1000][2];

    PARAM param;
    param.win_size = 11;
    param.block_size = 5;
    param.min_dist = 2;
    param.quality = 0.001;
    param.roiCorr = 7;
    param.umbral = 100;
    param.umbralcc = 0.75;
    param.use_harris = 1;


    cout << endl << "Calculando puntos en correspondencias..." << endl;

    CalcularPuntosCorrespondencias(img1, img2, &points1, &points2, param, &numPuntos);

    CvMat * lines1 = cvCreateMat(3, numPuntos, CV_32FC1);
    CvMat * lines2 = cvCreateMat(3, numPuntos, CV_32FC1);


    CvPoint p;

    for(int i=0; i<numPuntos; i++)
    {
        p.x = cvGetReal2D(points1, 0, i);
        p.y = cvGetReal2D(points1, 1, i);

        cvCircle(img1, p, 3, CvScalar());

        p.x = cvGetReal2D(points2, 0, i);
        p.y = cvGetReal2D(points2, 1, i);

        cvCircle(img2, p, 3, CvScalar());
    }



    // Hallamos la matriz fundamental
    cvFindFundamentalMat(points1, points2, F, CV_FM_RANSAC, 2);

    cout << endl <<"La Matriz fundamental (F) hallada es: " << endl;
    mostrarMatriz(F);


    // Hallamos las lineas epipolares
    cvComputeCorrespondEpilines(points2, 2, F, lines1);
    cvComputeCorrespondEpilines(points1, 1, F, lines2);


    float ep1x, ep1y, ep2x, ep2y;

    calcularPuntoCorte(lines1, 0, 1, &ep1x, &ep1y);
    calcularPuntoCorte(lines2, 0, 1, &ep2x, &ep2y);

    cout << endl << "El epipolo izquierdo es (" << ep1x << ", " << ep1y << ")." << endl;
    cout << endl << "El epipolo derecho es (" << ep2x << ", " << ep2y << ")." << endl;

    // Ahora tenemos que pintar las lineas, para lo que antes hallaremos dos puntos de cada linea

    for(int i=0; i<numPuntos; i++)
    {
        ptsLineas1[i][0].x = 0;
        ptsLineas1[i][0].y = obtenerPuntoLinea(lines1, 0, i);
        ptsLineas1[i][1].x = img1->width;
        ptsLineas1[i][1].y = obtenerPuntoLinea(lines1, img1->width, i);


        ptsLineas2[i][0].x = 0;
        ptsLineas2[i][0].y = obtenerPuntoLinea(lines2, 0, i);
        ptsLineas2[i][1].x = img2->width;
        ptsLineas2[i][1].y = obtenerPuntoLinea(lines2, img2->width, i);
    }

    // Pintamos las líneas en las imágenes

    for(int i=0; i<numPuntos; i++)
    {
        cvLine(img1, ptsLineas1[i][0], ptsLineas1[i][1], CvScalar());
        cvLine(img2, ptsLineas2[i][0], ptsLineas2[i][1], CvScalar());
    }


    cout << endl;
    cout << "Mostramos las líneas epipolares en cada imagen, junto con los puntos en correspondencia usados. " << endl;
    cout << "(para pasar de imagen pulsar una tecla)" << endl;

    pintaI("Primera", img1);
    pintaI("Segunda", img2);

    // Ahora debemos calcular la matriz de proyección asociada a cada cámara

    CvMat * Pi = cvCreateMat(3, 4, CV_32FC1);

    cvSetReal2D(Pi, 0, 0, 1);
    cvSetReal2D(Pi, 0, 1, 0);
    cvSetReal2D(Pi, 0, 2, 0);
    cvSetReal2D(Pi, 0, 3, 0);

    cvSetReal2D(Pi, 1, 0, 0);
    cvSetReal2D(Pi, 1, 1, 1);
    cvSetReal2D(Pi, 1, 2, 0);
    cvSetReal2D(Pi, 1, 3, 0);

    cvSetReal2D(Pi, 2, 0, 0);
    cvSetReal2D(Pi, 2, 1, 0);
    cvSetReal2D(Pi, 2, 2, 1);
    cvSetReal2D(Pi, 2, 3, 0);


    CvMat * Pd = cvCreateMat(3, 4, CV_32FC1);

    CvMat * ex = cvCreateMat(3, 3, CV_32FC1);

    cvSetReal2D(ex, 0, 0, 0);
    cvSetReal2D(ex, 0, 1, -ep1x);
    cvSetReal2D(ex, 0, 2, ep1y);

    cvSetReal2D(ex, 0, 0, ep1x);
    cvSetReal2D(ex, 0, 1, 0);
    cvSetReal2D(ex, 0, 2, -ep1x);

    cvSetReal2D(ex, 0, 0, -ep1y);
    cvSetReal2D(ex, 0, 1, ep1x);
    cvSetReal2D(ex, 0, 2, 0);

    CvMat * resultado = cvCreateMat(3, 3, CV_32FC1);

    cvGEMM(ex, F, 1, NULL, 1, resultado);


    for(int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
        {
            cvSetReal2D(Pd, i, j, cvGetReal2D(resultado, i, j));
        }
    }

    cvSetReal2D(Pd, 0, 3, ep2x);
    cvSetReal2D(Pd, 1, 3, ep2y);
    cvSetReal2D(Pd, 2, 3, 0);


    cout << endl;
    cout << "Las matrices de proyeccion halladas son las siguientes: " << endl << endl;
    mostrarMatriz(Pi);
    cout << endl << endl;
    mostrarMatriz(Pd);

    // Ya hemos encontrado las matrices de proyección, ahora calculamos la distancia ortogonal media

    float media = 0;

    for(int i=0; i<numPuntos; i++)
    {
        media += distanciaPuntoRecta(lines1, i, cvGetReal2D(points1, 0, i), cvGetReal2D(points1, 1, i));
        media += distanciaPuntoRecta(lines2, i, cvGetReal2D(points2, 0, i), cvGetReal2D(points2, 1, i));
    }

    media /= (numPuntos*2);

    cout << endl << "El error medio es: " << media << endl;
}







int main(int argc, char ** argv)
{
    cout << "PRACTICA 4: ESTIMACION DE GEOMETRIA EPIPOLAR" << endl << endl;
    cout << "RUBEN AGUILAR BECERRA - 25602111Y" << endl;

    cout << endl << "APARTADO 1" << endl;
    cout << "Resultados obtenidos para el conjunto de imágenes basement[00-01].tif" << endl;
    practica4_A();

    system("pause");

    cout << endl << "Resultados obtenidos para el conjunto de imágenes Vmort[1-2].pgm" << endl;
    Practica4_A_Vmort();


    cout << "APARTADO 2" << endl;
    cout << "Resultados obtenidos para el conjunto de imágenes basement[00-01].tif" << endl;
    Practica4_B();

    system("pause");

    cout << endl << "Resultados obtenidos para el conjunto de imágenes Vmort[1-2].pgm" << endl;
    Practica4_B_Vmort();


    cout << endl << endl << "PROGRAMA FINALIZADO " << endl;
    system("pause");

}







IplImage* pintarCuadraditos(IplImage * im, CvPoint2D32f* corners, int ncorners);


void primeraParte(char * imagen)
{
    printf("Primera parte: Hallar puntos de esquina por el metodo normal y por el de Harris\n\n");
    IplImage* im = cvLoadImage(imagen, CV_LOAD_IMAGE_GRAYSCALE);

    int count = 5000;
    CvPoint2D32f corners[5000];

    int countHarris = 5000;
    CvPoint2D32f cornersHarris[5000];


    puntosEsquina(im, corners, &count);



    puntosHarris(im, cornersHarris, &countHarris);


    printf("Mostramos los puntos que nos da el metodo normal \n\n");
    pintaI("Puntos Esquina", pintarCuadraditos(im, corners, count));

    printf("A continuación mostramos los que nos da el método de Harris \n\n");
    pintaI("Puntos Harris", pintarCuadraditos(im, cornersHarris, countHarris));
}


void puntosEsquina(IplImage * im, CvPoint2D32f * corners, int *cornerCount)
{

    IplImage* eig = cvCreateImage(cvSize(im->width, im->height), IPL_DEPTH_32F, 1);
    IplImage* temp = cvCreateImage(cvSize(im->width, im->height), IPL_DEPTH_32F, 1);

    double qualityLevel = 0.35;
    double minDistance = 6;

    cvGoodFeaturesToTrack(im, eig, temp, corners, cornerCount, qualityLevel, minDistance);
}



void puntosHarris(IplImage * im, CvPoint2D32f * corners, int *cornerCount)
{
    IplImage* eig = cvCreateImage(cvSize(im->width, im->height), IPL_DEPTH_32F, 1);
    IplImage* temp = cvCreateImage(cvSize(im->width, im->height), IPL_DEPTH_32F, 1);

    double qualityLevel = 0.08;
    double minDistance = 6;

    cvGoodFeaturesToTrack(im, eig, temp, corners, cornerCount, qualityLevel, minDistance, 0, 3, 1);
}


void segundaParte(char * im1, char * im2)
{
    char* name[2];

     IplImage* img[2];
     name[0]= im1;
     name[1]= im2;

     CvSize size;
     for(int ni=0; ni<2;ni++){
         if((img[ni]=cvLoadImage(name[ni],0))==NULL)
         { printf("error de lectura\n"); exit(0); }

         if(1){
             cvNamedWindow(name[ni],1);
             cvShowImage(name[ni],img[ni]);
             cvWaitKey(0);
             cvDestroyWindow(name[ni]);
         }

     }
             //definimos las imagenes auxiliares de cvGoodFeaturesToTrack

         size=cvGetSize(img[0]);
         IplImage* eigAux=cvCreateImage(size, IPL_DEPTH_32F, 1 );
         IplImage* tempAux=cvCreateImage(size, IPL_DEPTH_32F, 1);


         //reservamos memoria para guardar los puntos esquina de cada una de las imagenes
             // definir matrices de tamaño MAXCORNERS
         CvPoint2D32f* esq[2];

         esq[0] = (CvPoint2D32f*)cvAlloc(sizeof(CvPoint2D32f) * MAXCORNERS);
         esq[1] = (CvPoint2D32f*)cvAlloc(sizeof(CvPoint2D32f) * MAXCORNERS);

         int nesq0=MAXCORNERS;
         int nesq1=MAXCORNERS;

         int win_size=3; //cte a usar en cvFindCornerSubPix

         double min_dist=7; // cte a usar en cvGoodFeaturesToTrack
         double quality=0.01;  // cte a usar en cvGoodFeaturesToTrack

         cvGoodFeaturesToTrack(img[0], eigAux, tempAux, esq[0], &nesq0, quality, min_dist, 0, 3, 1);



         cvFindCornerSubPix(img[0], esq[0], nesq0, cvSize(win_size,win_size), cvSize(-1,-1), cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03));         //  ultimos parametros de esta llamada
                        //cvSize(win_size,win_size), cvSize(-1,-1),
                        //cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03));
         cvGoodFeaturesToTrack( img[1], eigAux, tempAux, esq[1], &nesq1, quality, min_dist, 0, 3, 1);
         cvFindCornerSubPix(img[1], esq[1], nesq1, cvSize(win_size,win_size), cvSize(-1,-1), cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03));

         //pintaI("Primera", pintarCuadraditos(img[0], esq[0], nesq0));

         //pintaI("Segunda", pintarCuadraditos(img[1], esq[1], nesq1));



         ////////////////////////////////////////////////////////////////////////////////////////////////
         ////////////////////////////////////////////////////////////////////////////////////////////////
         ////////////////////////////////////////////////////////////////////////////////////////////////
         ////////////////////////////////////////////////////////////////////////////////////////////////
         ////////////////////////////////////////////////////////////////////////////////////////////////

        //comenzamos aqui el calculo de la semejanza entre los puntos de cada imagen


         double coorVal[MAXCORNERS]={0};

         CvRect roi;
         int UMBRAL=3;  // umbral para la busqueda
         int anchoROI=11, altoROI=11 ;
         CvMat * vmatch = cvCreateMat(1, 1, CV_32FC1);  //matriz de salida de CvMatchTemplate()

         float valor;



         CvMat result=cvMat(1,1,CV_32FC1,&vmatch);

         //creamos la matriz que almacenara los resultados
         CvMat* corresp=cvCreateMat(nesq0, nesq1, CV_32FC1);
         cvZero(corresp);

             //en este doble bucle vamos calculando los valores de semejanza
             // y vamos descartando los que no nos interesan


         for(int y=0; y<nesq0; y++){
            roi=calculaROI(esq[0][y] ,anchoROI, altoROI,size);          //Calculamos el ROI
            if(roi.x==-1) continue;                                     // SI el ROI no es válido pasamos a la siguiente iteración
            cvSetImageROI(img[0],roi);                                  // Ajustamos el ROI de la imagen
            for(int x=0; x<nesq1; x++)                                  // Para cada punto Harris de la segunda imagen
            {
                valor = distancia(esq[0][y],esq[1][x] );                // Calculamos la distancia entre los puntos de Harris

                if( valor < UMBRAL ){                                   // SI la distancia está por debajo del umbral
                    roi=calculaROI(esq[1][x],anchoROI, altoROI,size);   // Calculamos el ROI para la segunda imagen
                    if(roi.x==-1) continue;                             // Si no es válido pasamos a la siguiente iteración
                    cvSetImageROI(img[1],roi);                          // Ajustamos el ROI de la imagen
                    cvMatchTemplate(img[0], img[1], vmatch, CV_TM_CCORR_NORMED);    // Calculamos el índice de correspondencia
                    valor=cvmGet(vmatch,0,0);
                    cvSetReal2D(corresp,y,x,valor);                     // Lo alamacenamos en la matriz de correspondencias
                    printf("%d-(%f, %f), %d-(%f,%f)\n", y, esq[0][y].x,esq[0][y].y,x,esq[1][x].x,esq[1][x].y);
                }
            }
        }

       cvResetImageROI(img[0]);
       cvResetImageROI(img[1]);

       CvMat header;
       double minVal,maxVal;
       CvPoint minLoc, maxLoc;
       int indx0[MAXCORNERS]={0};
       CvMat* fila;

    //en este bucle vamos calculando los maximos de cada fila
       for(int i=0; i<nesq0; i++)
       {
           fila=cvGetRow(corresp,&header,i);
           cvMinMaxLoc(fila, &minVal, &maxVal, &minLoc, &maxLoc);  // calculamos el valor maximo y la columna en donde esta
           indx0[i]= maxLoc.x;   // almacenamos el indice de la columna del maximo
           coorVal[i]=maxVal;  // almacenamos el valor de semejanza
       }

       //igual que antes pero para las columnas

       int indx1[MAXCORNERS]={0};

       CvMat* colum;
       for(int i=0; i<nesq1; i++)
       {
           colum = cvGetCol(corresp, &header,i);
           cvMinMaxLoc(colum, &minVal, &maxVal, &minLoc, &maxLoc);  // calculamos el valor maximo y la columna en donde esta
           indx1[i] = maxLoc.y; //almacenamos el indice de la fila del maximo
       }

    //cruzamos  las dos listas calculadas para ver que puntos son maximo de su fila y columna


    CvPoint indCorr[MAXCORNERS];  // matriz que almacena los indices de los puntos en correspondencias

    // bucle que busca por cada fila de la matriz si el valor maximo de dicha fila es tambien maximo
    // de su columna. Ademas añadimos que el valor de semejanza sea suficientemente alto.

    for(int i=0; i<nesq0; i++)
       if( (i == indx1[indx0[i]]) && (coorVal[i]>0.9))
           indCorr[i]=cvPoint(i,indx0[i]);  // guardamos ambos indices
       else indCorr[i]=cvPoint(-1,-1);

     CvPoint p;

    //pintamos los puntos uno a uno para poder ver cual se corresponde con cual

     int min;
     if(nesq0 > nesq1)
         min = nesq1;
     else
         min = nesq0;

     for(int i=0; i<min; i++)
         if (indCorr[i].x!=-1){
             printf("(%d,%d)\n", indCorr[i].x, indCorr[i].y);
             p=cvPoint(cvRound( esq[0][indCorr[i].x].x),cvRound(esq[0][indCorr[i].x].y));
             printf("(%d,%d)\n", p.x, p.y);
             cvLine(img[0],cvPoint(p.x-3,p.y),cvPoint(p.x+3,p.y),cvScalar(255,0,0),1,4);
             cvLine(img[0],cvPoint(p.x,p.y-3),cvPoint(p.x,p.y+3),cvScalar(255,0,0),1,4);
             cvShowImage(name[0],img[0]);
             p=cvPoint(cvRound( esq[1][indCorr[i].y].x ),cvRound(esq[1][indCorr[i].y].y ));
             printf("(%d,%d)\n", p.x, p.y);
             cvLine(img[1],cvPoint(p.x-3,p.y),cvPoint(p.x+3,p.y),cvScalar(255,0,0),1,4);
             cvLine(img[1],cvPoint(p.x,p.y-3),cvPoint(p.x,p.y+3),cvScalar(255,0,0),1,4);
             cvShowImage(name[1],img[1]);
             cvWaitKey(0);
         }

         cvReleaseImage(&eigAux);
         cvReleaseImage(&tempAux);

     for(int j=0;j<2; j++) {
         cvFree((void**)&esq[j]);
         cvDestroyWindow(name[j]);
         cvReleaseImage(&img[j]);
     }
}



void CalcularPuntosCorrespondencias(IplImage * im1, IplImage * im2, CvMat ** src1, CvMat ** dst1, PARAM param, int *num)
{
    CvSize size = cvGetSize(im1);
    IplImage* eigAux = cvCreateImage(size, IPL_DEPTH_32F, 1);
    IplImage* tempAux = cvCreateImage(size, IPL_DEPTH_32F, 1);

    CvPoint2D32f* esq[2];

    esq[0] = (CvPoint2D32f*)cvAlloc(sizeof(CvPoint2D32f) * MAXCORNERS);
    esq[1] = (CvPoint2D32f*)cvAlloc(sizeof(CvPoint2D32f) * MAXCORNERS);

    CvMat * src = *src1;
    CvMat * dst = *dst1;

    int nesq0 = MAXCORNERS;
    int nesq1 = MAXCORNERS;

    int win_size = param.win_size;
    int block_size = param.block_size;

    double min_dist = param.min_dist;
    double quality = param.quality;

    cvGoodFeaturesToTrack(im1, eigAux, tempAux, esq[0], &nesq0, quality, min_dist, 0, block_size, 1);
    cvFindCornerSubPix(im1, esq[0], nesq0, cvSize(win_size, win_size), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03));


    cvGoodFeaturesToTrack(im2, eigAux, tempAux, esq[1], &nesq1, quality, min_dist, 0, block_size, 1);
    cvFindCornerSubPix(im2, esq[1], nesq1, cvSize(win_size, win_size), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03));



    double coorVal[MAXCORNERS] = {0};

    CvSize sizeIm1 = cvGetSize(im1);
    CvSize sizeIm2 = cvGetSize(im2);

    CvRect roi;
    int UMBRAL = param.umbral;
    int anchoROI = param.roiCorr, altoROI = param.roiCorr;
    CvMat * vmatch;


    float valor;

    CvMat* corresp = cvCreateMat(nesq0, nesq1, CV_32FC1);
    cvZero(corresp);

    CvRect rect1, rect2;





    for(int y=0; y<nesq0; y++)
    {
        roi = calculaROI(esq[0][y], anchoROI, altoROI, size);
        if(roi.x == -1) continue;
        cvSetImageROI(im1, roi);
        rect1 = cvGetImageROI(im1);
        for(int x=0; x<nesq1; x++)
        {
            valor = distancia(esq[0][y], esq[1][x]);

            if(valor < UMBRAL)
            {
                roi = calculaROI(esq[1][x], anchoROI, altoROI, size);
                if(roi.x == -1) continue;
                cvSetImageROI(im2, roi);
                rect2 = cvGetImageROI(im2);
                vmatch = cvCreateMat(rect1.height - rect2.height +1, rect1.width - rect2.width +1, CV_32FC1);

                //printf("<%d, %d>, <%d, %d>, <%d, %d> \n",rect1.width, rect1.height, rect2.width, rect2.height, vmatch->width, vmatch->height);
                cvMatchTemplate(im1, im2, vmatch, CV_TM_CCORR_NORMED);
                valor = cvmGet(vmatch, 0, 0);
                cvSetReal2D(corresp, y, x, valor);
            }
        }
    }

    cvResetImageROI(im1);
    cvResetImageROI(im2);

    CvMat header;
    double minVal, maxVal;
    CvPoint minLoc, maxLoc;
    int indx0[MAXCORNERS] = {0};
    CvMat* fila;

    for(int i=0; i<nesq0; i++)
    {
        fila = cvGetRow(corresp, &header, i);
        cvMinMaxLoc(fila, &minVal, &maxVal, &minLoc, &maxLoc);
        indx0[i] = maxLoc.x;
        coorVal[i] = maxVal;
    }

    int indx1[MAXCORNERS] = {0};

    CvMat* colum;
    for(int i=0; i<nesq1; i++)
    {
        colum = cvGetCol(corresp, &header, i);
        cvMinMaxLoc(colum, &minVal, &maxVal, &minLoc, &maxLoc);
        indx1[i] = maxLoc.y;
    }



    // Aqui aplicamos el cambio respecto a lo anterior, almacenamos las correspondencias en dos vectores de doubles


    CvPoint indCorr[MAXCORNERS];


    for(int i=0; i<nesq0; i++)
    {
        if((i == indx1[indx0[i]]) && (coorVal[i]>param.umbralcc))
            indCorr[i] = cvPoint(i, indx0[i]);
        else
            indCorr[i] = cvPoint(-1,-1);
    }


    int count = 0;

    for(int i=0; i<nesq0; i++)
    {
        if(indCorr[i].x != -1)
        {
            count++;
        }
    }

    src = cvCreateMat(2, count, CV_32FC1);
    dst = cvCreateMat(2, count, CV_32FC1);

    count = 0;

    for(int i=0; i<nesq0; i++)
    {
        if(indCorr[i].x != -1)
        {
            cvmSet(src, 0, count, esq[0][indCorr[i].x].x);
            cvmSet(src, 1, count, esq[0][indCorr[i].x].y);
            cvmSet(dst, 0, count, esq[1][indCorr[i].y].x);
            cvmSet(dst, 1, count, esq[1][indCorr[i].y].y);
            count++;
        }
    }

    *num = count;
    **src1 = *src;
    **dst1 = *dst;
}



void terceraParte(char * im1, char * im2)
{
     char* name[3];

     IplImage* img[3];
     name[0]= im1;
     name[1]= im2;
     name[2]= im1;


     CvSize size;
     for(int ni=0; ni<3;ni++){
         if((img[ni]=cvLoadImage(name[ni],0))==NULL)
         { printf("error de lectura\n"); exit(0); }
     }
     int ancho = img[0]->height;
     int alto = img[0]->width;

     IplImage * velX = cvCreateImage(cvSize(img[0]->width, img[0]->height), 32, 1);
    IplImage * velY = cvCreateImage(cvSize(img[0]->width, img[0]->height), 32, 1);

    float distanciaPuntos;
    double dx, dy;

    cvCalcOpticalFlowLK(img[0], img[1], cvSize(5, 5), velX, velY);

         for(int i=0; i<img[0]->height; i+=3)
         {
             for(int j=0; j<img[0]->width; j+=3)
             {
                 dx = j + cvGetReal2D(velX, i, j);
                 dy = i + cvGetReal2D(velY, i, j);
                 distanciaPuntos = distancia(cvPoint2D32f(j,i), cvPoint2D32f(dx, dy));

                 if(!(distanciaPuntos < 2 || distanciaPuntos > 7))
                    cvLine(img[0], cvPoint(j,i), cvPoint(j+cvGetReal2D(velX, i, j), i+cvGetReal2D(velY, i, j)), cvScalar(0));
             }
         }

    printf("Flujo óptico hallado con el método de Lucas-Kanade\n\n");

    pintaI(name[0], img[0]);

    int BS = 4;

    CvSize sizeE = cvSize((img[2]->width)- BS, (img[2]->height)- BS);


     IplImage * velXi = cvCreateImage(sizeE, 32, 1);
    IplImage * velYi = cvCreateImage(sizeE, 32, 1);

    cvCalcOpticalFlowBM(img[2], img[1], cvSize(BS,BS) ,cvSize(1,1), cvSize(10,10), 0, velXi, velYi);

             for(int i=0; i<img[2]->height - BS; i+=3)
         {
             for(int j=0; j<img[2]->width - BS; j+=3)
             {
                 dx = j + cvGetReal2D(velXi, i, j);
                 dy = i + cvGetReal2D(velYi, i, j);
                 distanciaPuntos = distancia(cvPoint2D32f(j,i), cvPoint2D32f(dx, dy));

                 if(!(distanciaPuntos < 2 || distanciaPuntos > 7))
                    cvLine(img[2], cvPoint(j,i), cvPoint(j+cvGetReal2D(velXi, i, j), i+cvGetReal2D(velYi, i, j)), cvScalar(0));
             }
         }

    printf("FLujo óptico hallado con el método de Block-Matching\n\n");
    pintaI(name[2], img[2]);
}

void pintaI(char * name, IplImage * im)
{
    cvNamedWindow(name, 1);
    cvShowImage(name, im);
    cvWaitKey();
    cvDestroyWindow(name);
}




IplImage* pintarCuadraditos(IplImage * im, CvPoint2D32f* corners, int ncorners)
{
    IplImage* color = cvCreateImage(cvSize(im->width, im->height), IPL_DEPTH_8U, 3);

    cvCvtColor(im, color, CV_GRAY2RGB);

    CvScalar escalar = cvScalar(0.0, 0.0, 255);

    for(int i=0; i< ncorners; i++)
    {
        cvRectangle(color, cvPoint(corners[i].x - 2, corners[i].y+2), cvPoint(corners[i].x +2, corners[i].y -2), escalar);
    }

    return color;
}
