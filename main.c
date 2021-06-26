#include "ch.h"
#include "hal.h"

#include "chprintf.h"
#include "lsm303dlhc.h"
#include "l3gd20.h"
#include <stdlib.h>
#include <math.h>

#define cls(chp)  chprintf(chp, "\033[2J\033[1;1H")
#define FEATURE_NUMBER                  6
#define NUMAXES                         3
#define TIMEWINDOW                      256
#define NEXTPOW2                        256
#define PI                              3.14159
#define DIM_SPETTRO                     129
#define UNIFORMSAMPLERATE               100
#define HALFNEXTPOW                     128
#define DATALENGTH                      13

#define dMin1                           24.77510
#define dMin2                           18736.18646
#define dMin3                           96798.53788
#define dMin4                           0.39062
#define dMin5                           24.45860
#define dMin6                           1

#define Range1                          1010.51845
#define Range2                          95907439.7167
#define Range3                          123228634.10775
#define Range4                          13.28125
#define Range5                          4976.08189
#define Range6                          3

#define INPUT_NEURONS                   6
#define HIDDEN_NEURONS                  10

#define NUMBER_OF_CLASSES               3

typedef struct {
  float re;
  float im;
}cmplx;



/*  Dichiarazioni funzioni utilizzate */
void extractFeature(float acccooked[][NUMAXES], float featureVector[]);
void insertionSort(float x[], int n);
void FFT(cmplx* f, int N, double d);
void transform(cmplx* f, int N);
void ordina( cmplx* f1, int N);
int logb2(int N);
void linspace(float x1, float x2,float f[]);
float absolute(cmplx *pro, int p);
float modulo(float n);
int findpeaks(float* dataOfInterest, int dataLength, int minDist);
void normalizzazione(float featureVector[]);
int classification(float featureVector[]);
void rounds(int class);

/*  Variabili globali  */
int class;
float featureVector[FEATURE_NUMBER];
float mag[TIMEWINDOW];
cmplx freqAccel[NEXTPOW2];
float y[TIMEWINDOW];
float spectrumAmpl[DIM_SPETTRO];
float dataOfInterest[DATALENGTH];
float f[DIM_SPETTRO] ;

/* Variabili globali funzioni */
cmplx f2[NEXTPOW2];
cmplx W[HALFNEXTPOW];
float p[DIM_SPETTRO];
float peaks[DATALENGTH];
float temp[NUMBER_OF_CLASSES];
/*===========================================================================*/
/* LSM303DLHC related.                                                       */
/*===========================================================================*/

/* LSM303DLHC Driver: This object represent an LSM303DLHC instance */
static LSM303DLHCDriver LSM303DLHCD1;
static float acccooked[TIMEWINDOW][NUMAXES];
static uint32_t i;
static const I2CConfig i2ccfg = {
  STM32_TIMINGR_PRESC(15U) |
  STM32_TIMINGR_SCLDEL(4U) | STM32_TIMINGR_SDADEL(2U) |
  STM32_TIMINGR_SCLH(15U)  | STM32_TIMINGR_SCLL(21U),
  0,
  0
};

static const LSM303DLHCConfig lsm303dlhccfg = {
  &I2CD1,
  &i2ccfg,
  NULL,
  NULL,
  LSM303DLHC_ACC_FS_4G,
  LSM303DLHC_ACC_ODR_100Hz,
#if LSM303DLHC_USE_ADVANCED
  LSM303DLHC_ACC_LP_DISABLED,
  LSM303DLHC_ACC_HR_DISABLED,
  LSM303DLHC_ACC_BDU_BLOCK,
  LSM303DLHC_ACC_END_LITTLE,
#endif
  NULL,
  NULL,
  LSM303DLHC_COMP_FS_1P3GA,
  LSM303DLHC_COMP_ODR_30HZ,
#if LSM303DLHC_USE_ADVANCED
  LSM303DLHC_COMP_MD_BLOCK
#endif
};
/*===========================================================================*/
/* L3GD20 related.                                                           */
/*===========================================================================*/

/* L3GD20 Driver: This object represent an L3GD20 instance.*/
static L3GD20Driver L3GD20D1;
//static float gyrocooked[TIMEWINDOW][L3GD20_GYRO_NUMBER_OF_AXES];

//static char axisIDL3GD2[L3GD20_GYRO_NUMBER_OF_AXES] = {'X', 'Y', 'Z'};
static uint32_t i;
static const SPIConfig spicfg = {
  FALSE,
  NULL,
  GPIOE,
  GPIOE_L3GD20_CS,
  SPI_CR1_BR | SPI_CR1_CPOL | SPI_CR1_CPHA,
  0
};
static L3GD20Config l3gd20cfg = {
  &SPID1,
  &spicfg,
  NULL,
  NULL,
  L3GD20_FS_250DPS,
  L3GD20_ODR_760HZ,
#if L3GD20_USE_ADVANCED
  L3GD20_BDU_CONTINUOUS,
  L3GD20_END_LITTLE,
  L3GD20_BW3,
  L3GD20_HPM_REFERENCE,
  L3GD20_HPCF_8,
  L3GD20_LP2M_ON,
#endif
};
/*===========================================================================*/
/* Generic code.                                                             */
/*===========================================================================*/
static BaseSequentialStream* chp = (BaseSequentialStream*)&SD1;

float hid[HIDDEN_NEURONS] ;
float biasIn[HIDDEN_NEURONS] =  {-1.8486, -1.8581, -2.2676, 0.7520, 2.5815,26.7100, -0.7314,19.5950, -3.2430, 2.4198};
float biasOut = 0;
float inWeight[HIDDEN_NEURONS][INPUT_NEURONS] = {   {1.5959, 0.5955, -0.2238, 0.7404, 0.1728, 0.4936},
                                                    {0.7982, 0.5793, -0.4276, 0.9827, 0.3509, 1.3825},
                                                    {1.5587, 1.1355, -0.1110, 1.4362, 0.4777, 1.1838},
                                                    {0.8023, 3.2567, -0.2357, -1.2797, 0.5100, 1.4964},
                                                    {2.4238, 4.2294, 1.8709, -0.5626, 1.6323, 0.1413},
                                                    {16.6221, -0.1011, -17.7758, -0.0407, 45.1135, 0.7451},
                                                    {0.8445, 1.1650, -0.2489, 2.2080, -1.0471, -0.5741},
                                                    {17.1708, 30.9565, 13.5933, -5.2940, 8.2378, -0.6335},
                                                    {-1.6427, 0.8993, -0.5852, -0.0040, -3.0167, -0.1275},
                                                    {0.9599, 0.2055, -0.7421, -0.4017, -0.4994, 0.9576}};

float hidWeight[HIDDEN_NEURONS] = {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6667, 0.0000, 0.3333, 0.0000, 0.0000};
float classes[NUMBER_OF_CLASSES] = {26.0836, 32.4873, 40.4838};


int main(void) {

  halInit();
  chSysInit();

  /* Activates the serial driver 1 using the driver default configuration.*/
  sdStart(&SD1, NULL);

 /* LSM303DLHC Object Initialization.*/
  lsm303dlhcObjectInit(&LSM303DLHCD1);

  /* Activates the LSM303DLHC driver.*/
  lsm303dlhcStart(&LSM303DLHCD1, &lsm303dlhccfg);

  /* L3GD20 Object Initialization.*/
  l3gd20ObjectInit(&L3GD20D1);
  /* Activates the L3GD20 driver.*/
  l3gd20Start(&L3GD20D1, &l3gd20cfg);

  /* Inizializzazione matrice "acccoocked" a zero */
  for(int i = 0; i < TIMEWINDOW; i++){
    for( int j = 0; j < NUMAXES; j++) {
      acccooked[i][j] = 0;
    }
  }
  while(true) {

    /* Funzione shift della matrice "acccoocked */
    for(int i = TIMEWINDOW - 1; i >0; i--) {
      for( int j = 0; j < NUMAXES; j++) {
        acccooked[i][j] = acccooked[i - 1][j];
        }
      }
    lsm303dlhcAccelerometerReadCooked(&LSM303DLHCD1, acccooked[0]);

    /* Clear dei led */
    palClearLine(LINE_LED3_RED);
    palClearLine(LINE_LED4_BLUE);
    palClearLine(LINE_LED5_ORANGE);
    palClearLine(LINE_LED6_GREEN);
    palClearLine(LINE_LED7_GREEN);
    palClearLine(LINE_LED8_ORANGE);
    palClearLine(LINE_LED9_BLUE);
    palClearLine(LINE_LED10_RED);

    /* Extract Features dei valori della matrice "acccoocked" */
    extractFeature(acccooked, featureVector);
    /* Normalizzazione delle features */
    normalizzazione(featureVector);
    /* Classificazione delle features */
    class=classification(featureVector);
    /* Scelta dei led da accendere in base al valore di class */
    rounds(class);
    chThdSleepMilliseconds(5);

   /* Stampa delle features e della classe rilevate dalla scheda */
  /* for(int i=0; i<6;i++)
      chprintf(chp,"%f;",featureVector[i]);

   chprintf(chp,"%d;",class);
   chprintf(chp,"\n");
   cls(chp);
  */
  }
  lsm303dlhcStop(&LSM303DLHCD1);
  l3gd20Stop(&L3GD20D1);
}

void rounds(int class){
  /* IDLE */
  if(class == 0) {
    palToggleLine(LINE_LED3_RED);
    palToggleLine(LINE_LED4_BLUE);
    palToggleLine(LINE_LED5_ORANGE);
    palToggleLine(LINE_LED6_GREEN);
    palToggleLine(LINE_LED7_GREEN);
    palToggleLine(LINE_LED8_ORANGE);
    palToggleLine(LINE_LED9_BLUE);
    palToggleLine(LINE_LED10_RED);
  }
  /* WALK */
  else if(class == 1) {
    palToggleLine(LINE_LED5_ORANGE);
    palToggleLine(LINE_LED8_ORANGE);
  }
  /* RUN */
  else if( class == 2) {
    palToggleLine(LINE_LED6_GREEN);
    palToggleLine(LINE_LED7_GREEN);
  }
}

void extractFeature(float acccooked[][NUMAXES], float featureVector[]) {
  for(i = 0; i < FEATURE_NUMBER; i++){
    featureVector[i] = 0;
  }

  /* Calcolo Average */
  for ( i = 0; i < TIMEWINDOW; i++){
    mag[i] = sqrt(acccooked[i][1] * acccooked[i][1] + acccooked[i][2]*acccooked[i][2]) ;
    featureVector[0] = featureVector[0] + mag[i];
  }
  featureVector[0] = featureVector[0] / TIMEWINDOW;

  /* Ordinamento vettore "mag" */
  insertionSort(mag,TIMEWINDOW);

  /* Calcolo del 25° percentile */
  float p_raw, p_value;
  int p25, p75;



  p_raw = 0.25 * TIMEWINDOW;
  p25 = ceil(p_raw);
  /* Caso p25 decimale */
  if (p25 != p_raw){
    p25++;
    p_value = mag[p25];
  }
  /* Caso p25 intero */
  else
    p_value = (mag[p25] + mag[p25 + 1]) / 2;

  for (i = 0; i < p25; i++)
    featureVector[1] = featureVector[1] + pow(mag[i], 2);

  /* Calcolo 75 */
  p_raw = 0.75 * TIMEWINDOW;
  p75 = ceil(p_raw);

  /* Caso decimale */
  if (p75 != p_raw){
    p75++;
    p_value = mag[p75];
  }
  /* Caso intero */
  else
    p_value = (mag[p75] + mag[p75 + 1]) / 2;

  for (i = 0; i < p75; i++)
    featureVector[2] = featureVector[2] + pow(mag[i], 2);

  /* Calcolo media di Y */
  float mediaY=0;
  for (i = 0; i < TIMEWINDOW; i++)
    mediaY = mediaY + acccooked[i][1];
  mediaY = mediaY / TIMEWINDOW;

  /* Sottraggo ad Y la media */
  for (i = 0; i < TIMEWINDOW; i++) {
    y[i] = acccooked[i][1] - mediaY;
  }
  /* Rendo Y un vettore complesso */
  for (i = 0; i < NEXTPOW2; i++){
    if (i > TIMEWINDOW) {
      freqAccel[i].re=0;
      freqAccel[i].im=0;
    }
    else {
      freqAccel[i].re = y[i];
      freqAccel[i].im=0;
    }
  }

  /* Trasformata di Fourier veloce del vettore Y */
  FFT(freqAccel, NEXTPOW2, 1);

  /* Calcolo la frequenza di accelerazione */
  for (i = 0; i < NEXTPOW2; i++) {
    freqAccel[i].re = freqAccel[i].re / NEXTPOW2;
    freqAccel[i].im = freqAccel[i].im / NEXTPOW2;
  }

  float x1 = 0, x2 = 1;
  /* Calcola 129 intervalli equidistanti tra 0 e 1 */
  linspace(x1, x2,f);

  /* Calcolo l'ampiezza dello spettro */
  for (i = 0; i < DIM_SPETTRO; i++){
    spectrumAmpl[i] = 2 * absolute(freqAccel, i);
  }

  /* Calcolo Somma a 5Hz */
  float sum5Hz = 0;
  int n1 = ceil(NEXTPOW2 * 5 / (UNIFORMSAMPLERATE/ 2)) + 1;
  for (i = 0; i < n1; i++)
    sum5Hz = sum5Hz + spectrumAmpl[i];
  featureVector[4] = sum5Hz;

  /* Calcolo Max Frequenza */
  float max = spectrumAmpl[0];
  int index = 0;

  /* Ricerca del massimo */
  for (i = 1; i < DIM_SPETTRO; i++) {
    if (max < spectrumAmpl[i]) {
      max = spectrumAmpl[i];
      index = i;
    }
  }
  featureVector[3] = f[index];

  /* DATALENGTH = ceil(DIM_SPETTRO * (5 / UNIFORMSAMPLERATE/2)) */
  for (i = 0; i < DATALENGTH; i++)
     dataOfInterest[i] = spectrumAmpl[i];

  int minDistance = ceil(DIM_SPETTRO / UNIFORMSAMPLERATE);

  /* Ricerca dei Picchi */
  int npeaks = findpeaks(dataOfInterest, DATALENGTH, minDistance);
  featureVector[5]=npeaks;
}

int logb2(int N) {
  int k = N, i = 0;
  while(k){
    k >>= 1;
    i++;
  }
  return i - 1;
}

void insertionSort(float x[], int n) {
  float app;
  int i,j;
  for (i = 1; i < n; i++){
    app = x[i];
    for (j = i - 1; (j >= 0) && (x[j] > app); j--)
      x[j+1] = x[j];
    x[j + 1] = app;
  }
}

int check(int n) {
  return n > 0 && (n & (n - 1)) == 0;
}

int reverse(int N, int n) {
  int j, p = 0;
  for(j = 1; j <= logb2(N); j++) {
    if(n & (1 << (logb2(N) - j)))
      p |= 1 << (j - 1);
  }
  return p;
}

void ordina( cmplx* f1, int N) {

  int i,j;
  for ( i = 0; i < N; i++)
    f2[i].re = f1[reverse(N, i)].re;
  for ( j = 0; j < N; j++) {
    f1[j].re = f2[j].re;
    f1[j].im = f2[j].im;
  }
}

cmplx prodotto(cmplx a,cmplx b) {
  cmplx z;
  z.re= a.re*b.re -b.im*a.im;
  z.im=a.im*b.re + b.im*a.re;
  return z;
}

void transform(cmplx* f, int N) {
  int i,j;
  int n = 1;
  int a = N / 2;

  ordina(f, N);
  W[1].re=1*cos(2*PI/N) ;
  W[1].im=+1*sin(-2*PI/N);
  W[0].re = 1;
  W[0].im = 0;

  for( i = 2; i < N / 2; i++) {
    W[i].re = 1*cos(2*PI*i/N);
    W[i].im=1*sin(-2*PI*i/N);
   }

  for(j = 0; j < logb2(N); j++) {
    for( i = 0; i < N; i++) {
      if(!(i & n)) {
        cmplx tmp ,Temp;
        tmp.re = f[i].re;
        tmp.im = f[i].im;
        Temp = prodotto(W[(i * a) % (n * a)] , f[i + n]);

        f[i].re = tmp.re + Temp.re;
        f[i].im = tmp.im + Temp.im;
        f[i + n].re = tmp.re - Temp.re;
        f[i + n].im = tmp.im - Temp.im;
      }
    }
    n *= 2;
    a = a / 2;
  }
}

float absolute(cmplx *pro, int p) {
  float pro1 = pro[p].re;
  float pro2 = pro[p].im;
  float prova3 = sqrt(pow(pro1,2)+pow(pro2,2));
  return prova3;
}

float modulo(float n) {
    if (n < 0)
        n = n - (2 * n);
    return n;
}

void FFT(cmplx* f, int N, double d){
  int i;
  transform(f, N);
  for (i = 0; i < N; i++){
    f[i].re =f[i].re* d;
    f[i].im =f[i].im* d;
  }
}

void linspace(float x1, float x2,float f[]) {

  int i;
  float inter = (x1 + x2) / (DIM_SPETTRO - 1);

  p[0] = x1;
  for (i = 1; i < DIM_SPETTRO; i++) {
    float temp = p[i-1];
    p[i] = temp + inter;
  }
  for (i = 0; i < DIM_SPETTRO; i++)
    f[i] = UNIFORMSAMPLERATE / 2 * p[i];
}

int findpeaks(float* dataOfInterest, int dataLength, int minDist) {
  int peaksNumber = 0;
  int i,j = 0;

  for ( i = 1; i < dataLength - 1; i++){
    if (dataOfInterest[i] > dataOfInterest[i - 1]
                          && dataOfInterest[i] > dataOfInterest[i + 1]){
      peaksNumber++;
      peaks[j] = dataOfInterest[i];
      j++;
      i = i + minDist;
     }
   }
  return peaksNumber;
}

void normalizzazione(float featureVector[]) {
    featureVector[0]=(featureVector[0]-dMin1)/Range1;
    featureVector[1]=(featureVector[1]-dMin2)/Range2;
    featureVector[2]=(featureVector[2]-dMin3)/Range3;
    featureVector[3]=(featureVector[3]-dMin4)/Range4;
    featureVector[4]=(featureVector[4]-dMin5)/Range5;
    featureVector[5]=(featureVector[5]-dMin6)/Range6;
}

int classification(float featureVector[]) {
    int i, j ;
    float out=0;
    float min;
    int fine  = 0 ;


    for(i = 0 ; i < HIDDEN_NEURONS ; i ++) {
      hid[i]=0;
    }
    for(i = 0 ; i < HIDDEN_NEURONS ; i ++) {
        for(j = 0 ; j < INPUT_NEURONS; j++)
            hid[i] = hid[i] + inWeight[i][j]*featureVector[j];
        hid[i] = hid[i] + biasIn[i] ;
    }
    /* RELU */
     for(i = 0; i < HIDDEN_NEURONS; i ++) {
        if(hid[i] < 0)
            hid[i] = 0 ;
    }

    for(j = 0 ; j < HIDDEN_NEURONS; j++)
      out = out + hidWeight[j]*hid[j] ;
    out = out + biasOut;

   /* chprintf(chp,"%f;",out);
    chprintf(chp,"\n");
    cls(chp);*/
    for(i = 0 ; i < NUMBER_OF_CLASSES ; i ++) {
      temp[i] = out - classes[i];
      temp[i] = modulo(temp[i]);
    }

    min = temp[0];
    for(i = 1 ; i < NUMBER_OF_CLASSES ; i ++)
      if(min > temp[i]) {
        min = temp[i];
        fine = i ;
      }
   return fine;
}
