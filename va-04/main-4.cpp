#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
typedef unsigned int uint;

pair<vector<Point2f>, vector<Point2f> >  readPoints();
pair<Mat, Mat> epipolesFromFundamentalMat(Mat f);
pair<Mat, Mat> cameraMatsFromFundamentalMat(Mat f);
Mat drawEpilines(Mat im, vector<Point3f> epilines);
double averageReprojectionError(pair<vector<Point3f>, vector<Point3f> >epilines, pair<vector<Point2f>, vector<Point2f> >points);
void sift(Mat &img1, vector<KeyPoint> &keypoints);
pair<vector<Point2f>, vector<Point2f> > computeMatching(Mat &img1, Mat &img2, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2);
void uno();
void dos();

int main() {
    uno();
    dos();
    return 0;
}

pair<Mat, Mat> epipolesFromFundamentalMat(Mat f)
{
    Mat eigenValues, eigenVectors;
    double min, max;
    Point2i minLoc;

    eigen(f.t()*f, eigenValues, eigenVectors);
    minMaxLoc(abs(eigenValues), &min, &max, &minLoc);
    Mat e1;
    eigenVectors.row(minLoc.y).copyTo(e1);
    e1 /= e1.at<double>(0, 2);

    eigen(f*f.t(), eigenValues, eigenVectors);
    minMaxLoc(abs(eigenValues), &min, &max, &minLoc);
    Mat e2;
    eigenVectors.row(minLoc.y).copyTo(e2);
    e2 /= e2.at<double>(0, 2);

    return pair<Mat, Mat>(e1, e2);
}

Mat drawEpilines(Mat im, vector<Point3f> epilines)
{
    Mat aux_im(im);
    for (uint i=0; i<epilines.size(); i++){
        Point3d epiline = epilines[i];
        Point2i p1, p2;
        p1.x = 0;
        p1.y = (-p1.x*epiline.x - epiline.z)/epiline.y;
        p2.x = aux_im.cols;
        p2.y = (-p2.x*epiline.x - epiline.z)/epiline.y;
        clipLine(Size(aux_im.cols, aux_im.rows), p2, p1);
        line(aux_im, p2, p1, Scalar(1));
    }
    return aux_im;
}

/*******************************************************************************************************************************************
 *
 *
 *
 *
 *  REVISAR, estan las camaras realmente o la inversa de alguna?
 *
 *
 *
 */
pair<Mat, Mat> cameraMatsFromFundamentalMat(Mat f)
{
    pair<Point3d, Point3d> epipoles = epipolesFromFundamentalMat(f);
    //Matriz P, canonica
    Mat p1 = (Mat_<double>(3, 4) << 1, 0, 0, 0,  0, 1, 0, 0,   0, 0, 1, 0);
    //Matriz P' = [ [e']_x·f | e']
    Mat p2 = Mat::zeros(3, 4, f.type());
    Mat e2_x(3, 3, f.type());
    e2_x.at<double>(0, 1) = -epipoles.second.z;
    e2_x.at<double>(0, 2) =  epipoles.second.y;
    e2_x.at<double>(1, 0) =  epipoles.second.z;
    e2_x.at<double>(1, 2) = -epipoles.second.x;
    e2_x.at<double>(2, 0) = -epipoles.second.y;
    e2_x.at<double>(2, 1) =  epipoles.second.x;
    Mat e2xf = e2_x * f;
    Mat e2 = (Mat_<double>(3, 1) << epipoles.second.x, epipoles.second.y, epipoles.second.z);

    for (int i = 0; i < 3; i++){
        Mat p2_i(p2, Rect(0, i, 3, 1));
        e2xf.row(i).copyTo(p2_i);
    }

    p2.col(3).at<double>(0, 0) = epipoles.second.x;
    p2.col(3).at<double>(1, 0) = epipoles.second.y;
    p2.col(3).at<double>(2, 0) = epipoles.second.z;

    cout << "\n\nReconstruccion de F a partir de P y P':\n\n";
    cout << "f = [e']_x·P'·P+ =\n\n";
    Mat pinvert;
    invert(p1, pinvert, DECOMP_SVD);

    Mat r = e2_x * p2 * p1.inv(DECOMP_SVD);

    r /= r.at<double>(2, 2);
    cout << r << endl;
    return pair<Mat, Mat>(p1, p2);
}

double averageReprojectionError(pair<vector<Point3f>, vector<Point3f> > epilines, pair<vector<Point2f>, vector<Point2f> > points)
{
    double error = 0;

    for(uint i = 0; i < points.first.size(); i++){
        error += abs(epilines.first[i].x * points.first[i].x+ epilines.first[i].y * points.first[i].y + epilines.first[i].z)
                    / sqrt(epilines.first[i].x * epilines.first[i].x + epilines.first[i].y * epilines.first[i].y);
        error += abs(epilines.second[i].x * points.second[i].x+ epilines.second[i].y * points.second[i].y+ epilines.second[i].z)
                    / sqrt(epilines.second[i].x * epilines.second[i].x + epilines.second[i].y * epilines.second[i].y);
    }
    return error/(double)(points.first.size() + points.second.size());
}

void uno()
{
    cout << "########################################################\n";
    cout << "###################### Apartado 1 ######################\n";
    cout << "########################################################\n\n";
    //a. Leer los  puntos en correspondencias  de  las imágenes Vmort[*].pgm
    pair<vector<Point2f>, vector<Point2f> > points = readPoints();
    Mat im1 = imread("imagenes/Vmort1.pgm");
    Mat im2 = imread("imagenes/Vmort2.pgm");

    //b. Estimar la matriz F sin RANSAC usando el algoritmo de los 8 puntos. Usar findFundamentallMat()
    Mat fMat = findFundamentalMat(points.first, points.second, CV_FM_8POINT);
    cout <<"Matriz fundamental:\nF =\n\n" << fMat << endl;

    //c. Dibujar las lineas epipolares sobre ambas imágenes. Usar computeCorrespondEpilines(), clipLine() y line().
    pair<vector<Point3f>, vector<Point3f> > epilines;
    computeCorrespondEpilines(points.first,  1, fMat, epilines.second);
    computeCorrespondEpilines(points.second, 2, fMat, epilines.first);

    Mat im1_lines = drawEpilines(im2, epilines.first);
    Mat im2_lines = drawEpilines(im1, epilines.second);

    Mat imCompuesta(max(im1_lines.rows, im2_lines.rows), im1_lines.cols + im2_lines.cols, im1_lines.type());
    Mat fragmentoCompuesta1(imCompuesta, Range(0, im1_lines.rows), Range(0, im1_lines.cols));
    Mat fragmentoCompuesta2(imCompuesta, Range(0, im2_lines.rows), Range(im1_lines.cols, im1_lines.cols + im2_lines.cols));

    im2_lines.copyTo(fragmentoCompuesta1);
    im1_lines.copyTo(fragmentoCompuesta2);
    cout << "\nMostrando lineas epipolares, pulse una tecla para continuar\n";
    imshow("Lineas epipolares A", imCompuesta);
    waitKey();
    destroyWindow("Lineas epipolares A");

    //d. Calcular los epipolos y la matriz de proyección (P) asociada a cada cámara.
    cout << "Epipolos:\n";
    pair<Mat, Mat> epipoles = epipolesFromFundamentalMat(fMat);
    cout << "\te  = " << epipoles.first << endl;
    cout << "\te' = " << epipoles.second << endl;
    pair<Mat, Mat> cameraMats = cameraMatsFromFundamentalMat(fMat);
    cout << "\nPar de camaras canonico:\n";
    cout << "P = \n\n";
    cout << cameraMats.first << endl << endl;
    cout << "P' = \n\n";
    cout << cameraMats.second << endl << endl;
    //e. Verificar la bondad de la F estimada calculando la distancia ortogonal  media entre lineas epipolares y puntos correspondientes.
    cout <<"Error medio = " <<  averageReprojectionError(epilines, points) << endl;
}

void dos()
{
    cout << "########################################################\n";
    cout << "###################### Apartado 2 ######################\n";
    cout << "########################################################\n\n";
    cout << "Entiendo que hay error en el enunciado de este apartado porque se pide dos veces lo mismo, apartado c y d, asi que repito lo que se pide en el primer apartado.\n\n";

    pair<Mat, Mat> ims;
    ims.first  = imread("imagenes/Vmort1.pgm");
    ims.second = imread("imagenes/Vmort2.pgm");

    //ims.first = imread("imagenes/basement00.tif");
    //ims.second = imread("imagenes/basement01.tif");
    cvtColor(ims.first, ims.first, CV_RGB2GRAY);
    cvtColor(ims.second, ims.second, CV_RGB2GRAY);
    cout << "Calculando correspondencias...\n";
    pair <vector<KeyPoint>, vector<KeyPoint> > points;
    sift(ims.first, points.first);
    sift(ims.second, points.second);
    pair<vector<Point2f>, vector<Point2f> > matches = computeMatching(ims.first, ims.second, points.first, points.second);

    //calculo de outliers
    Mat status;
    findFundamentalMat(matches.first, matches.second, CV_FM_RANSAC, 1, 0.99999, status);
    //eliminacion de los outliers dados por RANSAC
    pair<vector<Point2f>, vector<Point2f> > ransacInliersMatches;
    for (int i = 0; i < status.rows; i++){
        if ((int)status.at<uchar>(i, 0)){
            ransacInliersMatches.first.push_back (matches.first[i]);
            ransacInliersMatches.second.push_back(matches.second[i]);
        }
    }

    Mat fMat = findFundamentalMat(ransacInliersMatches.first, ransacInliersMatches.second, CV_FM_8POINT, 0, 0, status);
    cout <<"Matriz fundamental:\nF =\n\n" << fMat << endl;

    //calculo de las rectas epipolares
    pair<vector<Point3f>, vector<Point3f> > epilines;
    computeCorrespondEpilines(matches.first,  1, fMat, epilines.second);
    computeCorrespondEpilines(matches.second, 2, fMat, epilines.first);

    Mat im1_lines = drawEpilines(ims.first, epilines.first);
    Mat im2_lines = drawEpilines(ims.second, epilines.second);

    Mat imCompuesta(max(im1_lines.rows, im2_lines.rows), im1_lines.cols + im2_lines.cols, im1_lines.type());
    Mat fragmentoCompuesta1(imCompuesta, Range(0, im1_lines.rows), Range(0, im1_lines.cols));
    Mat fragmentoCompuesta2(imCompuesta, Range(0, im2_lines.rows), Range(im1_lines.cols, im1_lines.cols + im2_lines.cols));
    im2_lines.copyTo(fragmentoCompuesta2);
    im1_lines.copyTo(fragmentoCompuesta1);
    imshow("Lineas epipolares B", imCompuesta);
    waitKey();
    destroyWindow("Lineas epipolares B");


    //d. Calcular los epipolos y la matriz de proyección (P) asociada a cada cámara.
    cout << "\n\nEpipolos:\n";
    pair<Mat, Mat> epipoles = epipolesFromFundamentalMat(fMat);
    cout << "\te  = " << epipoles.first << endl;
    cout << "\te' = " << epipoles.second << endl;
    pair<Mat, Mat> cameraMats = cameraMatsFromFundamentalMat(fMat);
    cout << "\nPar de camaras canonico:\n";
    cout << "P = \n\n";
    cout << cameraMats.first << endl << endl;
    cout << "P' = \n\n";
    cout << cameraMats.second << endl << endl;

    cout << "Error medio por punto = " << averageReprojectionError(epilines, matches) << endl;
}

pair<vector<Point2f>, vector<Point2f> >  readPoints()
{
    ifstream in("imagenes/Vmort1.LMSMatch");
    float x, y;
    pair<vector<Point2f>, vector<Point2f> > points;

    while(!in.eof()){
        in >> x;
        in >> y;
        points.first.push_back(Point2f(x, y));
    }
    in.close();

    points.second = vector<Point2f>();
    in.open("imagenes/Vmort2.LMSMatch");
    while(!in.eof()){
        in >> x;
        in >> y;
        points.second.push_back(Point2f(x, y));
    }
    in.close();
    return points;
}

void sift(Mat &img1, vector<KeyPoint> &keypoints)
{
    SIFT::CommonParams commParams = SIFT::CommonParams();
    commParams.nOctaves = 4;
    commParams.nOctaveLayers = 3;
    SIFT::DetectorParams detectorParams = SIFT::DetectorParams();
    detectorParams.threshold = 0.08; //parametro que fija la cantidad de respuestas
    detectorParams.edgeThreshold = 8;
    SIFT detector = SIFT(commParams, detectorParams);
    Mat mask;
    detector(img1, mask, keypoints);
}

pair<vector<Point2f>, vector<Point2f> > computeMatching(Mat &img1, Mat &img2, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2)
{
    SiftDescriptorExtractor extractor;
    Mat descriptors1, descriptors2;
    extractor.compute(img1, keypoints1, descriptors1);
    extractor.compute(img2, keypoints2, descriptors2);
    BruteForceMatcher<L2<float> > matcher;
    vector<DMatch> matches1_2, matches2_1;
    matcher.match(descriptors1, descriptors2, matches1_2);
    matcher.match(descriptors2, descriptors1, matches2_1);
    pair<vector<Point2f>, vector<Point2f> > matches;
    vector<DMatch> dmatchFiltrado;
    double maxDistance = 90;
    for (uint i=0; i < matches1_2.size(); i++){
        if (matches1_2[i].distance > maxDistance){
            continue;
        }
        pair<Point2f, Point2f> match1_2 = pair<Point2f, Point2f>(keypoints1[matches1_2[i].queryIdx].pt, keypoints2[matches1_2[i].trainIdx].pt);
        for (uint j=0; j < matches2_1.size(); j++){
            if (matches2_1[j].distance > maxDistance){
                continue;
            }
            pair<Point2f, Point2f> match2_1 = pair<Point2f, Point2f>(keypoints1[matches2_1[j].trainIdx].pt, keypoints2[matches2_1[j].queryIdx].pt);
            if (match1_2.first == match2_1.first && match1_2.second == match2_1.second){
                if (dmatchFiltrado.empty() || (matches.first.back() != match1_2.first || matches.second.back() != match1_2.second)){
                    dmatchFiltrado.push_back(matches1_2[i]);
                    matches.first.push_back(match1_2.first);
                    matches.second.push_back(match1_2.second);
                }
            }
        }
    }

    Mat img3;
    drawMatches(img1, keypoints1, img2, keypoints2, dmatchFiltrado, img3);
    imshow("Correspondencias", img3);
    waitKey();
    destroyWindow("Correspondencias");
    return matches;
}

/*
 * Estimar la matriz fundamental a partir de un conjunto de puntos homogéneos en correspondencia.
PROGRAMACIÓN:
Cuestionario de preguntas: 3.5 puntos
Dados dos conjunto de puntos {xi} y {xi'} en correspondencias, entre dos imágenes,
calcular la matriz F que verifica la ecuación xi'TFx=0.
1.Calcular F usando el Algoritmo de 8-puntos  (1.5 puntos):
    a. Leer los  puntos en correspondencias  de  las imágenes Vmort[*].pgm
    b. Estimar la matriz F sin RANSAC usando el algoritmo de los 8 puntos. Usar findFundamentallMat()
    c. Dibujar las lineas epipolares sobre ambas imágenes. Usar computeCorrespondEpilines(), clipLine() y line() .
    d. Calcular los epipolos y la matriz de proyección (P) asociada a cada cámara.
    e. Verificar la bondad de la F estimada calculando la distancia ortogonal  media entre lineas epipolares y puntos
        correspondientes. Mostrar el valor medio por punto.

2.Calcular F usando el Algoritmo 8 puntos+RANSAC  (1  puntos ):
    a. Obtener  puntos en correspondencias sobre las imágenes Vmort[*].pgm de forma automática usando las funciones SIFT
        ya dadas.
    b. Calcular F por el algoritmo de los 8 puntos + RANSAC (usar un valor razonable para el error de RANSAC)
    c. Verificar la bondad de la F estimada calculando la distancia ortogonal  media entre lineas epipolares y puntos
        correspondientes.
    d. Pintar las lineas epipolares resultantes de la matriz F .
    e. Verificar la bondad de la F estimada calculando la distancia ortogonal  media entre lineas epipolares y puntos
        correspondientes. Mostrar el valor medio por punto.

DATOS: En el fichero datos se adjuntan dos conjuntos de imágenes y una de ellas con ficheros con los valores de puntos
    en correspondencias.
El programa deberá  dar de salida la estimación de la matriz F por filas seguida de las coordenadas de los epipolos
de ambas imágenes, y las matrices de proyección asociadas (por filas).
 */