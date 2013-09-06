#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


vector<Mat> leerImagenes ();
void pintaImagen(Mat imagen, std::string ventana);
vector<pair<bool, vector<Point2f> > > ejercicio1(const vector<Mat> &imagenes);
vector<pair<Mat, vector<Point2f> > > imagenesAptas(vector<Mat> imagenes, vector<pair<bool, vector<Point2f> > > datosImagenes);
vector<vector<Point2f> > ejercicio2(vector<pair<Mat, vector<Point2f> > > imagenesYdatos);
pair<double, pair<Mat, Mat> > ejercicio3(vector<vector<Point2f> > puntos, Size dimensionImagenes, bool corregirRadial = true, bool corregirTangencial = true);
void ejercicio4(vector<Mat> imagenes, Mat camara, Mat coeficientes);

void pintaImagen(Mat imagen, std::string ventana){
    namedWindow(ventana);
    Mat imagen2 = Mat(Size(512, 512), imagen.type());
    imshow(ventana, imagen);
    waitKey();
    destroyWindow(ventana);
}

vector<Mat> leerImagenes (){
    vector<Mat> imagenes;
    int n_imagenes = 25;
    char ruta[50];
    for(int i=0; i < n_imagenes; i++){
        sprintf(ruta, "./imagenes/Image%d.tif", i+1);
        imagenes.push_back(imread(ruta));
        cvtColor(imagenes.back(), imagenes.back(), CV_RGB2GRAY);
    }
    return imagenes;
}

//Determina las imagenes aptas
vector<pair<bool, vector<Point2f> > > ejercicio1(const vector<Mat> &imagenes){
    vector<pair<bool, vector<Point2f> > > aptas;
    vector<Point2f> puntos;
    for(uint i=0; i< imagenes.size(); i++){
        aptas.push_back(pair<bool, vector<Point2f> >());
        aptas.back().first = findChessboardCorners(imagenes.at(i), Size(12, 12), puntos);
        if (aptas.back().first){
            aptas.back().second = puntos;
            cout << "Imagen " << i << " apta\n";
        }else{
            cout << "Imagen " << i << " no apta\n";
            aptas.back().second = vector<Point2f>();
        }
    }
    return aptas;
}

//Rellena un vector solo con las imagenes aptas, se podria unificar con el ejercicio 1
vector<pair<Mat, vector<Point2f> > > imagenesAptas(vector<Mat> imagenes, vector<pair<bool, vector<Point2f> > > datosImagenes){
    vector<pair<Mat, vector<Point2f> > > imagenesValidas;
    for (uint i = 0; i < imagenes.size(); i++){
        if (datosImagenes[i].first){
            imagenesValidas.push_back(pair<Mat, vector<Point2f> >(imagenes[i], datosImagenes[i].second));
        }
    }
    cout << "n validas " << imagenesValidas.size() << endl;;
    return imagenesValidas;
}

vector<vector<Point2f> > ejercicio2(vector<pair<Mat, vector<Point2f> > > imagenesYdatos){
    vector<vector<Point2f> > puntosMejorados;
    for (uint i = 0; i < imagenesYdatos.size(); i++){
        cornerSubPix(imagenesYdatos[i].first, imagenesYdatos[i].second, Size(5, 5), Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100000, 0.00001));
        puntosMejorados.push_back(imagenesYdatos[i].second);
        cout << "Revisando la imagen apta " << i << endl;
        drawChessboardCorners(imagenesYdatos[i].first, Size(12, 12), puntosMejorados.back(), true);
        pintaImagen(imagenesYdatos[i].first, "Esquinas tras subPix");
    }

    return puntosMejorados;
}

pair<double, pair<Mat, Mat> > ejercicio3(vector<vector<Point2f> > puntos, Size dimensionImagenes, bool corregirRadial, bool corregirTangencial){
    vector<vector<Point3f> > mundos;
    mundos.push_back(vector<Point3f>());
    //genera la cuadricula
    for(int i = 0; i < 12; i++){
        for(int j = 0; j < 12; j++){
            mundos.back().push_back(Point3f(i*5, j*5, 0));
        }
    }

    //replica los datos de la cuadricula
    for (uint i=1; i<puntos.size(); i++){
        mundos.push_back(mundos.front());
    }

    //estima el valor inicial de la matriz de camara
    Mat valorInicial = initCameraMatrix2D(mundos, puntos, dimensionImagenes);

    vector<Mat> rotaciones, translaciones;
    Mat distCoeffs;
    for (uint i=0; i < puntos.size(); i++){
        rotaciones.push_back(Mat());
        translaciones.push_back(Mat());
        solvePnP(mundos[i], puntos[i], valorInicial, distCoeffs, rotaciones.back(), translaciones.back(), false);
    }

    distCoeffs.zeros(distCoeffs.size(), distCoeffs.type());

    //flags comunes
    int flags = CV_CALIB_USE_INTRINSIC_GUESS + CV_CALIB_FIX_PRINCIPAL_POINT;

    //no corregir el error tangencial
    if (!corregirTangencial){
        flags += CV_CALIB_ZERO_TANGENT_DIST;
    }

    //no corregir el error radial
    if (!corregirRadial){
        flags += CV_CALIB_FIX_K1 + CV_CALIB_FIX_K2 + CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5 + CV_CALIB_FIX_K6;
    }

    double error = calibrateCamera(mundos, puntos, dimensionImagenes, valorInicial, distCoeffs, rotaciones, translaciones, flags);

    return pair<double, pair<Mat, Mat> >(error, pair<Mat, Mat>(valorInicial, distCoeffs));
    /*
    * La salida del ejercicio 3 es tal que asi
    *
    * Error con ambas correcciones ----> 0.23798
    * Error con correccion radial -----> 0.25648
    * Error con correccion tangencial -> 1.2554
    * Error sin correcciones ----------> 1.28144
    *
    * Como se puede ver el error causado por la deformacion tangencial es despreciable en comparacion con el error
    * producido por la distorsion radial de las lentes, muy a menudo el error tangencial ni tan siquiera se corrige
    * Dado que el error sin correcciones es de 1.28144 (100%) los aportes de cada tipo de error al error total son de
    * Error tangencial   = Error sin correcciones - Error con correccion radial     = 0.02604  (~2.0321%)
    * Error radial       = Error sin correcciones - Error con correccion tangencial = 1.02469  (~79.98501%)
    * Error no corregido = Error sin correcciones - Error con ambas correcciones    = 0.23798  (~18.57129%)
    * Como se puede el error causado (y corregido) por la deformacion radial es 40 veces mayor que el error causado
    * (y corregido) por la deformacion tangencial.
    *
    * Aumentar la precision del calculo de cornerSubPix en varios ordenes de magnitud apenas a mejorado el error
    * corregido por la estimacion del error radial.
    *
    * Los errores mostrados corresponden a una ejecucion con un calculo a nivel de subpixel con menor precision.
    */
}

void ejercicio4(vector<Mat> imagenes, Mat camara, Mat coeficientes){
    for(uint i=0; i<imagenes.size(); i++){
        Mat imagenAux, imgCompuesta;
        undistort(imagenes[i], imagenAux, camara, coeficientes);
        drawMatches(imagenes[i], vector<KeyPoint>(), imagenAux, vector<KeyPoint>(), vector<DMatch>(),  imgCompuesta);
        pintaImagen(imgCompuesta, "");
    }

    return;
}

int main(void) {
    cout << "Leer Imagenes\n";
    vector<Mat> imagenes = leerImagenes();
    vector<Mat> imagenesOriginales;

    //Se copia porque drawChessboardCorners machaca las imagenes
    for(uint i=0; i<imagenes.size(); i++){
        imagenesOriginales.push_back(Mat());
        imagenes[i].copyTo(imagenesOriginales.back());
    }

    cout << "Ejercicio 1, reconocer imagenes no validas\n";
    vector<pair<bool, vector<Point2f> > > datosImagenes = ejercicio1(imagenes);
    vector<pair<Mat, vector<Point2f> > >  imagenesYdatosAptos = imagenesAptas(imagenes, datosImagenes);

    cout << "Ejercicio 2, determinar y mostrar los puntos a nivel de subPix\n";
    vector<vector<Point2f> > puntosMejorados =  ejercicio2(imagenesYdatosAptos);

    cout << "Ejercicio 3\n";
    pair<double, pair<Mat, Mat> >datosCalibracion1 = ejercicio3(puntosMejorados, imagenes.front().size(), true,  true);
    pair<double, pair<Mat, Mat> >datosCalibracion2 = ejercicio3(puntosMejorados, imagenes.front().size(), true,  false);
    pair<double, pair<Mat, Mat> >datosCalibracion3 = ejercicio3(puntosMejorados, imagenes.front().size(), false, true);
    pair<double, pair<Mat, Mat> >datosCalibracion4 = ejercicio3(puntosMejorados, imagenes.front().size(), false, false);

    cout <<"Error con ambas correcciones ----> " <<  datosCalibracion1.first << endl;
    cout <<"Error con correccion radial -----> " <<  datosCalibracion2.first << endl;
    cout <<"Error con correccion tangencial -> " <<  datosCalibracion3.first << endl;
    cout <<"Error sin correcciones ----------> " <<  datosCalibracion4.first << endl;

    cout << "Ejercicio 4\n";
    ejercicio4(imagenesOriginales, datosCalibracion1.second.first, datosCalibracion1.second.second);

    return 0;
}



/* Cuestionario  dela practica
 Cuestionario teórico - Práctica-3
==================================

1.- ¿Cuales son los elementos básicos de un modelo de cámara pin-hole? Justificar la respuesta.
- Plano de proyeccion: Es el plano sobre el que se proyecta la imagen
- Centro de la camara: Es el punto de fuga de la proyeccion.
- Eje principal: Recta que pasa por el centro de la camara y el plano de proyeccion
- Punto principal: Punto de corte del eje principal con el plano de proyeccion
- f: Distancia desde el  centro de la camara al punto principal
###########################################################################################################################################
2.- ¿Es la cámara pin-hole una cámara de proyección central? Justificar la respuesta.
Si, en una camara de proyeccion central todos los puntos pasan por un punto de fuga como ocurre en el centro de la camara en una camara tipo pin-hole.
###########################################################################################################################################
3.- ¿Que ecuaciones son las fundamentales para calcular la proyección de un punto del espacio en una imagen usando una cámara pìn-hole?. Justificar la respuesta.
La ecuacion es (x’, y’)^T = f / z (x, y)^T <== > x’ = f x / z, Y = f y / z,
se obtienen mediante semejanza de triangulos  o trigonometria basica tal que:
tan(Y) = y / z -> y’ = f * tan(Y) => y' = f * y / z
###########################################################################################################################################
4.- ¿Por que son necesarias las lentes de una cámara fotográfica? Justificar la respuesta.
Son necesarias para hacer converger la luz en un punto  y así  reducir el desenfoque propio del aumento de la apertura. Una apertura "grande" es necesaria para aumentar la cantidad de luz que recibe la camara y asi poder aumentar la velocidad de obturacion, lo que aumenta la velocidad con la que se realizan las fotos.
###########################################################################################################################################
5.- ¿Que relación existe entre la apertura de una cámara y su capacidad para obtener fotos enfocadas a distintas distancias?
Al reducir la apertura se reduce la cantidad de rayos (se reduce el circulo de confusion) que se reciben de cada punto de la escena y, de esta manera, los puntos fuera del plano enfocado aparecen mas nitidos.
###########################################################################################################################################
6.- ¿Porque es útil formular el modelo matematico de una cámara como una matriz que proyecta puntos en coordenadas homogéneas.
Porque de esta manera la camara se convierte en una aplicacion lineal. En coordenadas cartesianas se tiene que  (X, Y, Z ) - f -> (X / Z, Y / Z) que no es una aplicacion lineal, al transformarlo  en coordenadas homogeneas  (X / Z, Y / Z)^T  ⇒  (fx, fy, Z)^T se tiene una aplicacion f: (X, Y, Z) ->(fX, fY, Z) que si es lineal,
###########################################################################################################################################
7.- ¿Que tipo de cámara es una cámara finita? Justificar la respuesta.
----- No tengo nada claro a lo que se refiere esta pregunta -----
Una camara de finita es una camara de proyeccion central en la que todos los rayos pasan por el centro de la camara. Dado que PC = 0 se puede demostrar que C es el centro de la camara. Partiendo de la forma vectorial de la ecuacion de la recta que pasa por un punto cualquiera A y el centro de la camara C
X(L) = L·A + (1 - L) C
Aplicando la matriz de la camara P a los puntos de la recta tenemos que
x = P·X(L) = LPA + (1 - L) PC
Teniendo en cuenta que P·C = 0
x = LPA <=> x = PA (dado que P es invariante a escala)
Se concluye que todos los puntos  de la recta X(L) se proyectan sobre el mismo punto, esto implica que todas las lineas de proyeccion pasan por C
###########################################################################################################################################
8.- ¿Cualquier matriz 3x4 representa una cámara? Justificar la respuesta.
No, segun el tipo de camara:
Camaras finitas:
  Dado que la matriz camara puede descomponerse como P = KR [ I | -C] donde R es una matriz de rotacion  (y por tanto ortogonal) y K es
  triangular superior  luego se tiene que KR es similar a una factorizacion QR de una matriz M = KR. luego tambien se sabe que tiene
  det(M) = det(K)*det(R) , det(R) != 0 por ser matriz de rotacion y det(K)  != 0 por construccion esto implica que det(M) != 0,
  por tanto la matriz M triangular superior de la matriz P es no singular y por tanto P tiene rango 3.

Camaras generales:
  Si se elimina la restriccion aplicada a las camaras finitas debe mantenerse que rango(P) = 3 en otro caso la proyeccion a traves de P
  sera una linea o un punto, no un plano y por lo tanto no una imagen 2D
###########################################################################################################################################
9.- ¿Es posible calcular el centro de cualquier cámara a partir de su matriz? Justificar la respuesta
  Si, el centro de la camara es el punto C tal que P·C = 0

10.- ¿Cuantas correspondencias son necesarias para estimar la matriz de una cámara ? Justificar la respuesta
  La situacion es la misma que en la estimacion de las homografias, solo cambia la dimension, de 3x3 a 3x4, por lo tanto se tienen 12
  grados de libertad, 11 en realidad pues se puede ignorar la escala. Cada pareja en correspondencia da tres ecuaciones pero una de ellas
  es linealmente dependiente de las otras por cada pareja se obtienen dos ecuaciones LI por lo tanto se necesitan 5’5 parejas es decir, 6.
###########################################################################################################################################
11 .-¿ Que información de una escena 3D se pierde siempre en una fotografía de la misma? Justificar la respuesta
  - La profundidad, puesto que tras proyectar todos los puntos que estan sobre un mismo rayo son proyectados sobre el mismo punto de la
  imagen.
  - Los angulos, aunque se mantiene la colinealidad, las rectas paralelas se cortan en los puntos de fuga. Esto puede demostrase calculando
  la proyeccion de dos puntos de dos rectas paralelas y comprobando que la ecuacion de las rectas que generan las proyecciones de esos
  puntos se cortan.
###########################################################################################################################################
12.- ¿Cual es la relación entre una cámara finita y una cámara afín? Justificar la respuesta
  Dada una camara finita, si esta se mueve hacia atrás a la vez que se escala para mantener la misma escena encuadrada cuando la distancia
  sea infinita la finita deriva en una camara con la ultima fila de la forma (0, 0, 0, k), o (0, 0, 0, 1) si se divide por k, que es la
  forma de una camara afin.
###########################################################################################################################################
13.- ¿De que depende el error de proyeccion de una cámara afín? Justificar la respuesta
  x_proyeccion = P_0·X  = K (x y d_0 + Delta)^T
  x_afin = P_inf · X = K (x y d_0)^T
Reescribiendo las matrices teniendo en cuenta que
    K = | K_2x2  x_0|
    | 0^T     1 |
se tiene que
        x_proyeccion = |K_2x2 x + (d_0+ Delta)*x0|
                       |         d_0 + Delta     |

        x_afin = |K_2x2 x + d_0 x_0|
                 |     d_0         |
Y el error sera:
  x_afin - x_proyeccion = Delta/d_0 (x_proyeccion - x_0)

Donde Delta es la profundidad media del objeto y d_0 la distancia media al objeto es decir, el error sera mayor cuanto mayor sea el factor
Delta/d_0 o, equivalentemente, cuanto menos despreciable sea la profundidad media del objeto con respecto a la distancia al mismo.
Este error se debe a que la camara afin realiza una proyeccion paralela de la zona de la escena tras del plano situado a distancia d_0 del
centro de la camara, sobre el que ademas realiza la proyeccion, con lo que se pierde el aporte de la profundida de dicha zona a la escena
###########################################################################################################################################
14.- ¿Seria posible rectificar la deformacion de las lentes de las imagenes que han sido descartadas en el proceso de calibracion de la camara?
  Si, los parametros de calibracion de la camara son propios de esta y no de las imagenes usadas para calibrarla, las transformaciones
  producidas por la camara estan presetentes en todas las imagenes tomadas por esta, no solo en las aptas para ser usadas como calibracion.

*/