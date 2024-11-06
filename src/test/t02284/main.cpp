#include <iostream>
#include <math.h>
#include <string>
/* do not use >=, or compiler may not know to optimize this with a simple (x &
 * 0x80...0). */
#define fabs(x) __ev_fabs(x)
const unsigned int count_per_degree = 60 * 60 * 1024;
const double degree_per_count = 1.0 / (60 * 60 * 1024);

#define EARTH_R 6378137.0

static double __ev_fabs(double x) { return x > 0.0 ? x : -x; }

void transform(double x, double y, double *lat, double *lng) {
  double xy = x * y;
  double absX = sqrt(fabs(x));
  double xPi = x * M_PI;
  double yPi = y * M_PI;
  double d = 20.0 * sin(6.0 * xPi) + 20.0 * sin(2.0 * xPi);

  *lat = d;
  *lng = d;

  *lat += 20.0 * sin(yPi) + 40.0 * sin(yPi / 3.0);
  *lng += 20.0 * sin(xPi) + 40.0 * sin(xPi / 3.0);

  *lat += 160.0 * sin(yPi / 12.0) + 320 * sin(yPi / 30.0);
  *lng += 150.0 * sin(xPi / 12.0) + 300.0 * sin(xPi / 30.0);

  *lat *= 2.0 / 3.0;
  *lng *= 2.0 / 3.0;

  *lat += -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * xy + 0.2 * absX;
  *lng += 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * xy + 0.1 * absX;
}

static int outOfChina(double lat, double lng) {
  if (lng < 72.004 || lng > 137.8347) {
    return 1;
  }
  if (lat < 0.8293 || lat > 55.8271) {
    return 1;
  }
  return 0;
}

static void delta(double lat, double lng, double *dLat, double *dLng) {
  if ((dLat == NULL) || (dLng == NULL)) {
    return;
  }
  const double ee = 0.00669342162296594323;
  transform(lng - 105.0, lat - 35.0, dLat, dLng);
  double radLat = lat / 180.0 * M_PI;
  double magic = sin(radLat);
  magic = 1 - ee * magic * magic;
  double sqrtMagic = sqrt(magic);
  *dLat = (*dLat * 180.0) / ((EARTH_R * (1 - ee)) / (magic * sqrtMagic) * M_PI);
  *dLng = (*dLng * 180.0) / (EARTH_R / sqrtMagic * cos(radLat) * M_PI);
}



void wgs2gcj(double wgsLat, double wgsLng, double *gcjLat, double *gcjLng) {
  if ((gcjLat == NULL) || (gcjLng == NULL)) {
    return;
  }
  if (outOfChina(wgsLat, wgsLng)) {
    *gcjLat = wgsLat;
    *gcjLng = wgsLng;
    return;
  }
  double dLat, dLng;
  delta(wgsLat, wgsLng, &dLat, &dLng);
  *gcjLat = wgsLat + dLat;
  *gcjLng = wgsLng + dLng;
}

void gcj2wgs(double gcjLat, double gcjLng, double *wgsLat, double *wgsLng) {
  if ((wgsLat == NULL) || (wgsLng == NULL)) {
    return;
  }
  if (outOfChina(gcjLat, gcjLng)) {
    *wgsLat = gcjLat;
    *wgsLng = gcjLng;
    return;
  }
  double dLat, dLng;
  delta(gcjLat, gcjLng, &dLat, &dLng);
  *wgsLat = gcjLat - dLat;
  *wgsLng = gcjLng - dLng;
}

bool MakeWgLongLat(double lon, double lat, unsigned int *wg_lng, unsigned int *wg_lat) {
  *wg_lng = (unsigned int)(lon * count_per_degree);
  *wg_lat = (unsigned int)(lat * count_per_degree);
  return true;
}

bool ParseWgLongLat(unsigned int wg_lng, unsigned int wg_lat, double *lon, double *lat) {
  *lon = wg_lng * degree_per_count;
  *lat = wg_lat * degree_per_count;
  return true;
}

unsigned int wgtochina_lb(int wg_flag, unsigned int wg_lng, unsigned int wg_lat, int wg_heit,
                          int wg_week, unsigned int wg_time, unsigned int *china_lng,
                          unsigned int *china_lat) {
  double lon, lat;
  ParseWgLongLat(wg_lng, wg_lat, &lon, &lat);
  double gcj_lon, gcj_lat;
  wgs2gcj(lat, lon, &gcj_lat, &gcj_lon);
  MakeWgLongLat(gcj_lon, gcj_lat, china_lng, china_lat);
  return 0;
}



int main()
{
  // 121.16331530017996 31.287102911608176 121.15506999138668 31.288027255529414
  double wgsLat,wgsLng = 0;
  auto to_string = [](double v){
 const int __n = 
      __gnu_cxx::__numeric_traits<double>::__max_exponent10 + 20;
  return __gnu_cxx::__to_xstring<std::string>(&std::vsnprintf, __n,
					   "%.9f", v);
              
  };
 
  gcj2wgs( 31.287102911608176,  121.16331530017996, &wgsLat,&wgsLng);
  std::cout << to_string(121.16331530017996) << "," << to_string(31.287102911608176) << "," << to_string( wgsLat) << "," <<to_string( wgsLng);
  return 0;
}