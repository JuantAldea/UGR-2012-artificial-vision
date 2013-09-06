#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void pintaImagen(std::string ventana, Mat imagen);
void computeMatching(Mat& img1, Mat& img2,vector<KeyPoint>& keypoints1,vector<KeyPoint>& keypoints2, vector<DMatch>& matches);
void sift(Mat& img1, vector<KeyPoint>& keypoints);
void show_matrix(Mat m);
Mat translationToCanvasCenter(Mat img, int canvas_rows, int canvas_cols);

void show_matrix(Mat m)
{
    for (int i = 0; i < m.cols; i++) {
        for (int j = 0; j < m.rows; j++) {
            cout << m.at<double>(i, j)/m.at<double>(2, 2) << ' ';
        }
        cout << endl;
    }
}

void pintaImagen(std::string ventana, Mat imagen)
{
    namedWindow(ventana);
    imshow(ventana, imagen);
    waitKey();
    destroyWindow(ventana);
}

void sift(Mat& img1, vector<KeyPoint>& keypoints)
{
    SIFT::CommonParams commParams = SIFT::CommonParams();
    commParams.nOctaves=4;
    commParams.nOctaveLayers=3;
    SIFT::DetectorParams detectorParams= SIFT::DetectorParams();
    detectorParams.threshold=0.06; // parametro que fija la cantidad de respuestas
    detectorParams.edgeThreshold=8;
    SIFT detector=SIFT(commParams,detectorParams);
    Mat mask;
    detector.operator()(img1, mask,keypoints);
}

void computeMatching(Mat& img1, Mat& img2,vector<KeyPoint>& keypoints1,vector<KeyPoint>& keypoints2, vector<DMatch>& matches)
{
    //const SIFT::DescriptorParams descriptorParams = SIFT::DescriptorParams();//variable no usada
    SiftDescriptorExtractor extractor;
    Mat descriptors1, descriptors2;
    extractor.compute(img1, keypoints1, descriptors1);
    extractor.compute(img2, keypoints2, descriptors2);
    BruteForceMatcher<L2<float> > matcher;
    matcher.match(descriptors1, descriptors2, matches);
}

void ejercicio1() {
    vector<Mat> imgs;
    vector<vector<Point2d> > matches;
    int n_images = 5;
    for (int i = 0; i < n_images; i++) {
        matches.push_back(vector<Point2d>());
        char pointsPath[50], imgsPath[50];
        sprintf(pointsPath, "imagenes/data%d.txt", i+1);
        sprintf(imgsPath, "imagenes/CalibIm%d.PNG", i+1);
        imgs.push_back(imread(imgsPath));
        cout << pointsPath << endl;
        cout << imgsPath << endl;
        ifstream input(pointsPath);
        for (int j = 0; j < 256; j++) {
            float x, y;
            input >> x;
            input >> y;
            matches.back().push_back(Point2d(x, y));
            j++;
        }
    }

    vector<Mat> homographies;
    for (uint i = 0; i < matches.size() - 1; i++) {
        homographies.push_back(Mat());
        homographies.back() = findHomography(matches[i], matches[i+1]);
    }

    vector<Mat> transformedImages;
    for (uint i = 0; i < homographies.size(); i++) {
        transformedImages.push_back(Mat());
        warpPerspective(imgs[i], transformedImages.back(), homographies[i], Size(imgs[i].cols*1.2, imgs[i].rows*1.2));
    }

    vector<Mat>rectifiedAndObjetiveImage;
    for (uint i = 0; i < transformedImages.size(); i++) {
        rectifiedAndObjetiveImage.push_back(Mat());
        drawMatches(transformedImages[i], vector<KeyPoint>(), imgs[i+1], vector<KeyPoint>(), vector<DMatch>(), rectifiedAndObjetiveImage.back());
        cout << "imagen " << i << endl;
    }

    for (uint i = 0; i < rectifiedAndObjetiveImage.size(); i++) {
        pintaImagen("Imagen i e i+1, i rectificada a i+1",rectifiedAndObjetiveImage[i]);
    }
}

void ejercicio2()
{
    //lectura de las imagenes
    char path[20];
    vector<Mat> imgs_gray;
    vector<Mat> imgs_color;

    int n_images = 8;
    for (int i=0; i < n_images; i++) {
        sprintf(path, "imagenes/fig7.9%c.PNG", (char)('a'+ i));
        cout << path << endl;
        Mat im = imread(path);
        imgs_color.push_back(im);
        cvtColor(im, im, CV_RGB2GRAY);
        imgs_gray.push_back(im);
    }

    //calculo de los puntos sift para cada imagen
    vector<vector<KeyPoint> > vectorKeypoints;
    for (uint i=0; i<imgs_gray.size(); i++) {
        vectorKeypoints.push_back(vector<KeyPoint>());
        sift(imgs_gray[i], vectorKeypoints.back());
    }

    //calculo de las correspondencias sift
    vector<vector<DMatch> > vectorMatches;
    for (uint i = 0; i < imgs_gray.size() - 1; i++) {
        vectorMatches.push_back(vector<DMatch>());
        computeMatching(imgs_gray[i], imgs_gray[i+1], vectorKeypoints[i], vectorKeypoints[i+1], vectorMatches.back());
    }

    //calculo de los puntos en correspondencia "real" para el calculo de las homografias
    vector<pair<vector<Point2f>,vector<Point2f> > > vectorImagePairsMatchedPoints;
    for (uint i = 0; i < vectorMatches.size(); i++) {
        vectorImagePairsMatchedPoints.push_back(pair<vector<Point2f>,vector<Point2f> >());
        vectorImagePairsMatchedPoints.back().first  = vector<Point2f>();
        vectorImagePairsMatchedPoints.back().second = vector<Point2f>();
        for (uint j = 0; j < vectorMatches[i].size(); j++) {
            vectorImagePairsMatchedPoints.back().first.push_back (vectorKeypoints[i][vectorMatches[i][j].queryIdx].pt);
            vectorImagePairsMatchedPoints.back().second.push_back(vectorKeypoints[i+1][vectorMatches[i][j].trainIdx].pt);
        }
    }


    //calculo de las homografias entre cada par consecutivo de imagenes
    vector<Mat> vectorHomographies;
    for (uint i = 0; i < vectorImagePairsMatchedPoints.size(); i++) {
        vectorHomographies.push_back(findHomography(vectorImagePairsMatchedPoints[i].first,
                                     vectorImagePairsMatchedPoints[i].second,
                                     CV_RANSAC));
    }

    vector<Mat> transformedImages;
    for (uint i=0; i < vectorHomographies.size(); i++) {
        transformedImages.push_back(Mat());
        warpPerspective(imgs_color[i], transformedImages.back(), vectorHomographies[i],  Size(imgs_color[i].cols * 1.2, imgs_color[i].rows * 1.2));
    }
    Mat imagenTrasteada;
    //warpPerspective(img1, imagenTrasteada, homography,  Size(img1.cols*2, img1.rows*2));

    //pinta cada par de imagenes adyacentes con y sin correspondencias sift.
    //tambien pinta la imagen rectificada y la original
    vector<Mat> vectorPairsMatchedImages;
    vector<Mat> vectorPairsRectifiedImages;
    vector<Mat> vectorPairsRectifiedAndOriginalImages;
    for (uint i = 0; i < imgs_color.size() - 1; i++) {
        vectorPairsMatchedImages.push_back(Mat());
        vectorPairsRectifiedImages.push_back(Mat());
        vectorPairsRectifiedAndOriginalImages.push_back(Mat());
        drawMatches(imgs_color[i], vectorKeypoints[i], imgs_color[i+1], vectorKeypoints[i+1], vectorMatches[i], vectorPairsMatchedImages.back());
        drawMatches(transformedImages[i], vector<KeyPoint>(), imgs_color[i+1], vector<KeyPoint>(),   vector<DMatch>(), vectorPairsRectifiedImages.back());
        drawMatches(transformedImages[i], vector<KeyPoint>(), imgs_color[i], vector<KeyPoint>(),   vector<DMatch>(), vectorPairsRectifiedAndOriginalImages.back());
    }

    //mostrar las imagenes
    for (uint i = 0; i < vectorPairsMatchedImages.size(); i++) {
        pintaImagen("Original y siguiente con correspondencias", vectorPairsMatchedImages[i]);
        pintaImagen("Rectificada y siguiente", vectorPairsRectifiedImages[i]);
        pintaImagen("Rectificada y original", vectorPairsRectifiedAndOriginalImages[i]);

    }
}

//es una copia exacta del 2 sin mas que cambiar la lectura de las imagenes
void ejercicio3()
{
    //lectura de las imagenes
    char path[20];
    vector<Mat> imgs_gray;
    vector<Mat> imgs_color;


    int n_images = 8; //+2
    for (int i=0; i < n_images; i++) {
        sprintf(path, "imagenes/mosaico00%d.jpg", i+2);
        cout << path << endl;
        Mat im = imread(path);
        imgs_color.push_back(im);
        cvtColor(im, im, CV_RGB2GRAY);
        imgs_gray.push_back(im);
    }

    //no me complico la vida para leer las que no tienen dos ceros
    Mat im = imread("imagenes/mosaico010.jpg");
    cout << "imagenes/mosaico010.jpg" << endl;
    imgs_color.push_back(im);
    cvtColor(im, im, CV_RGB2GRAY);
    imgs_gray.push_back(im);

    im = imread("imagenes/mosaico011.jpg");
    cout << "imagenes/mosaico011.jpg" << endl;
    imgs_color.push_back(im);
    cvtColor(im, im, CV_RGB2GRAY);
    imgs_gray.push_back(im);

    //calculo de los puntos sift para cada imagen
    vector<vector<KeyPoint> > vectorKeypoints;
    for (uint i=0; i<imgs_gray.size(); i++) {
        vectorKeypoints.push_back(vector<KeyPoint>());
        sift(imgs_gray[i], vectorKeypoints.back());
    }

    //calculo de las correspondencias sift
    vector<vector<DMatch> > vectorMatches;
    for (uint i = 0; i < imgs_gray.size() - 1; i++) {
        vectorMatches.push_back(vector<DMatch>());
        computeMatching(imgs_gray[i], imgs_gray[i+1], vectorKeypoints[i], vectorKeypoints[i+1], vectorMatches.back());
    }

    //calculo de los puntos en correspondencia "real" para el calculo de las homografias
    vector<pair<vector<Point2f>,vector<Point2f> > > vectorImagePairsMatchedPoints;
    for (uint i = 0; i < vectorMatches.size(); i++) {
        vectorImagePairsMatchedPoints.push_back(pair<vector<Point2f>,vector<Point2f> >());
        vectorImagePairsMatchedPoints.back().first  = vector<Point2f>();
        vectorImagePairsMatchedPoints.back().second = vector<Point2f>();
        for (uint j = 0; j < vectorMatches[i].size(); j++) {
            vectorImagePairsMatchedPoints.back().first.push_back (vectorKeypoints[i][vectorMatches[i][j].queryIdx].pt);
            vectorImagePairsMatchedPoints.back().second.push_back(vectorKeypoints[i+1][vectorMatches[i][j].trainIdx].pt);
        }
    }


    //calculo de las homografias entre cada par consecutivo de imagenes
    vector<Mat> vectorHomographies;
    for (uint i = 0; i < vectorImagePairsMatchedPoints.size(); i++) {
        vectorHomographies.push_back(findHomography(vectorImagePairsMatchedPoints[i].first,
                                     vectorImagePairsMatchedPoints[i].second,
                                     CV_RANSAC));
    }

    vector<Mat> transformedImages;
    for (uint i = 0; i < vectorHomographies.size(); i++) {
        transformedImages.push_back(Mat());
        warpPerspective(imgs_color[i], transformedImages.back(), vectorHomographies[i],  Size(imgs_color[i].cols * 1.2, imgs_color[i].rows * 1.2));
    }
    //pinta cada par de imagenes adyacentes con y sin correspondencias sift.
    //tambien pinta la imagen rectificada y la original
    vector<Mat> vectorPairsMatchedImages;
    vector<Mat> vectorPairsRectifiedImages;
    vector<Mat> vectorPairsRectifiedAndOriginalImages;
    for (uint i = 0; i < imgs_color.size() - 1; i++) {
        vectorPairsMatchedImages.push_back(Mat());
        vectorPairsRectifiedImages.push_back(Mat());
        vectorPairsRectifiedAndOriginalImages.push_back(Mat());
        drawMatches(imgs_color[i], vectorKeypoints[i], imgs_color[i+1], vectorKeypoints[i+1], vectorMatches[i], vectorPairsMatchedImages.back());
        drawMatches(transformedImages[i], vector<KeyPoint>(), imgs_color[i+1], vector<KeyPoint>(),   vector<DMatch>(), vectorPairsRectifiedImages.back());
        drawMatches(transformedImages[i], vector<KeyPoint>(), imgs_color[i], vector<KeyPoint>(),   vector<DMatch>(), vectorPairsRectifiedAndOriginalImages.back());
    }

    //mostrar las imagenes
    for (uint i = 0; i < vectorPairsMatchedImages.size(); i++) {
        pintaImagen("Original y siguiente con correspondencias", vectorPairsMatchedImages[i]);
        pintaImagen("Rectificada y siguiente", vectorPairsRectifiedImages[i]);
        pintaImagen("Rectificada y original",  vectorPairsRectifiedAndOriginalImages[i]);
    }
}

void ejercicio4()
{
    char path[20];
    vector<Mat> imgs_gray;
    vector<Mat> imgs_color;

    int n_images = 5;
    for (int i=0; i < n_images; i++) {

//#define fotos_escuela //para hacer el mosaico con las fotos desde la escuela
#ifndef fotos_escuela
        //sprintf(path, "imagenes/fig7.9%c.PNG", (char)('a'+ i));
	sprintf(path, "habitacion%d.jpg", i+1);
#else
        sprintf(path, "imagenes/mosaico00%d.jpg", i+2);

#endif
        cout << path << endl;
        Mat im = imread(path);
        imgs_color.push_back(im);
        cvtColor(im, im, CV_RGB2GRAY);
        imgs_gray.push_back(im);
    }

#ifdef fotos_escuela
    cout << "imagenes/mosaico010.jpg" << endl;
    Mat im = imread("imagenes/mosaico010.jpg");
    imgs_color.push_back(im);
    cvtColor(im, im, CV_RGB2GRAY);
    imgs_gray.push_back(im);

    cout << "imagenes/mosaico011.jpg" << endl;
    im = imread("imagenes/mosaico011.jpg");
    imgs_color.push_back(im);
    cvtColor(im, im, CV_RGB2GRAY);
    imgs_gray.push_back(im);
#endif

    //calculo de los puntos sift para cada imagen
    vector<vector<KeyPoint> > vectorKeypoints;
    for (uint i = 0; i < imgs_gray.size(); i++) {
      cout << "puntos shift " << i << endl;
        vectorKeypoints.push_back(vector<KeyPoint>());
        sift(imgs_gray[i], vectorKeypoints.back());

    }

    //calculo de las correspondencias sift
    vector<vector<DMatch> > vectorMatches;
    for (uint i = 0; i < imgs_gray.size() - 1; i++) {
      cout << "correspondencias shift " << i << endl;
        vectorMatches.push_back(vector<DMatch>());
        computeMatching(imgs_gray[i], imgs_gray[i+1], vectorKeypoints[i], vectorKeypoints[i+1], vectorMatches.back());

    }

    //calculo de los puntos en correspondencia "real" para el calculo de las homografias
    vector<pair<vector<Point2f>,vector<Point2f> > > vectorImagePairsMatchedPoints;
    for (uint i = 0; i < vectorMatches.size(); i++) {
        vectorImagePairsMatchedPoints.push_back(pair<vector<Point2f>,vector<Point2f> >());
        vectorImagePairsMatchedPoints.back().first  = vector<Point2f>();
        vectorImagePairsMatchedPoints.back().second = vector<Point2f>();
        for (uint j = 0; j < vectorMatches[i].size(); j++) {
            vectorImagePairsMatchedPoints.back().first.push_back (vectorKeypoints[i][vectorMatches[i][j].queryIdx].pt);
            vectorImagePairsMatchedPoints.back().second.push_back(vectorKeypoints[i+1][vectorMatches[i][j].trainIdx].pt);
        }
        cout << "validacion cruzada " << i << endl;
    }

    //calculo de las homografias entre cada par consecutivo de imagenes
    int axis_image = 1;
    int canvas_rows = imgs_gray[axis_image].rows * 2.5;
    int canvas_cols = imgs_gray[axis_image].cols * 6;
    Mat canvas(canvas_rows, canvas_cols, imgs_gray[axis_image].type());
    vector<Mat> vectorHomographies;
    for (uint i = 0; i < vectorImagePairsMatchedPoints.size(); i++) {
        vectorHomographies.push_back(findHomography(vectorImagePairsMatchedPoints[i].first,
                                     vectorImagePairsMatchedPoints[i].second,
                                     CV_RANSAC, 3));
	cout << "estimando homografias entre pares " << i << endl;
    }

    //AB BC CD DE EF FG GH CANVAS
    //0  1  2  3  4  5  6  7
    //A  B  C  D  E  F  G  H

    //translacion de la imagen eje al centro del canvas, con cutrez para descentrarla.
    Mat translation = translationToCanvasCenter(imgs_color[axis_image], canvas_rows, canvas_cols-imgs_gray[axis_image].rows *4);
    warpPerspective(imgs_color[axis_image], canvas, translation, Size(canvas_cols, canvas_rows));

    //Como las homografias estan calculadas llevando la imagen i-esima al plano de la i+1-esima
    //en las imagenes anteriores a la imagen eje se les aplica la h calculada sin mas,
    //en las imagenes que siguen a la imagen eje hay que aplicar la h inversa
    Mat transformation;
    translation.copyTo(transformation);
    for (int i = axis_image - 1; i>=0; i--) {
        transformation *= vectorHomographies[i];
        warpPerspective(imgs_color[i], canvas, transformation, Size(canvas_cols, canvas_rows),  INTER_LANCZOS4, BORDER_TRANSPARENT);
    }

    translation.copyTo(transformation);
    for (uint i = axis_image+1; i < imgs_color.size(); i++) {
        transformation *= vectorHomographies[i-1].inv();
        warpPerspective(imgs_color[i], canvas, transformation, Size(canvas_cols, canvas_rows),  INTER_LANCZOS4, BORDER_TRANSPARENT);
    }

    pintaImagen("Composicion", canvas);
}

Mat translationToCanvasCenter(Mat img, int canvas_rows, int canvas_cols)
{
    vector<Point2f> srcPoints;
    srcPoints.push_back(Point2f(0, 0));//(0,0)
    srcPoints.push_back(Point2f(img.cols - 1, 0));//(0,1)
    srcPoints.push_back(Point2f(img.cols - 1, img.rows - 1));//(1,1)
    srcPoints.push_back(Point2f(0, img.rows - 1));//(1,0)

    int canvas_center_row   = canvas_rows / 2;
    int canvas_center_col = canvas_cols / 2;

    int canvas_offset_row = canvas_center_row - img.rows / 2;
    int canvas_offset_col = canvas_center_col - img.cols / 2;

    vector<Point2f> dstPoints;

    dstPoints.push_back(Point2f(canvas_offset_col, canvas_offset_row));//(0,0)
    dstPoints.push_back(Point2f(canvas_offset_col + img.cols - 1, canvas_offset_row));//(0, 1)
    dstPoints.push_back(Point2f(canvas_offset_col + img.cols - 1, canvas_offset_row+img.rows - 1));//(1, 1)
    dstPoints.push_back(Point2f(canvas_offset_col, canvas_offset_row+img.rows-1));//(1, 0)

    return findHomography(srcPoints, dstPoints);
}

int main()
{
    cout << "\n\n======================= Ejercicio 1 =======================\n\n";
    //ejercicio1();
    cout << "\n\n======================= Ejercicio 2 =======================\n\n";
    //ejercicio2();
    cout << "\n\n======================= Ejercicio 3 =======================\n\n";
    //ejercicio3();
    cout << "\n\n======================= Ejercicio 4 =======================\n\n";
    ejercicio4();
    cout << endl << endl;
}


/*
 * A) Estimar  matrices de homografías  H a partir de los ficheros de puntos en correspondencias que se adjuntan
 *	 (data[1-5].txt). Cada fichero corresponde a una imagen (CalibIm[1-5].png).
 *	 Rectificar la imagen origen con la homografia calculada y compararla con la imagen destino.
 *	 Usar las funciones findHomography, warpPerspective. Mostrar los resultados( 1.5 punto)
 *
 * B)  Usar el método SIFT para calcular y establecer las correspondencias entre puntos relevantes
 * 	de las imágenes fig7.9[a-h]  (cada dos consecutivas).  Usar dichas correspondencias para calcular
 * 	las homografia entre las imágenes usando RANSAC. Mostrar las imágenes rectificadas y las originales. (1.5 puntos)
 *
 * C)  Hacer lo mismo que en el punto anterior  pero  con el conjunto de imágenes mosaico[2-11].  (1 puntos)
 *
 * D)  Estimar el mosaico generado a partir de las imágenes fig7.9[a-h]. Tomar como ejes del mosaico los definidos
 * 	por la imagen fig7.9d.  Usar las homografías que las relacionan. Mostrar el resultado. Opciones alternativas:
 * 	1. Mosaico de 2 imágenes (d y e): 1 punto
 * 	2. Mosaico de 3 imágenes ( c,d y e): 1.5 puntos
 * 	4. Mosaico total con todas imágenes : 2 puntos
*/
