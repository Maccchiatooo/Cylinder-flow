#include "lbm.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <fstream>

using namespace std;
using namespace sycl;
using namespace cl::sycl;

void LBM::Initialize()
{
    // index bound values for different cpu cores

    t[0] = 8.0 / 27.0;
    t[1] = 2.0 / 27.0;
    t[2] = 2.0 / 27.0;
    t[3] = 2.0 / 27.0;
    t[4] = 2.0 / 27.0;
    t[5] = 2.0 / 27.0;
    t[6] = 2.0 / 27.0;
    t[7] = 1.0 / 54.0;
    t[8] = 1.0 / 54.0;
    t[9] = 1.0 / 54.0;
    t[10] = 1.0 / 54.0;
    t[11] = 1.0 / 54.0;
    t[12] = 1.0 / 54.0;
    t[13] = 1.0 / 54.0;
    t[14] = 1.0 / 54.0;
    t[15] = 1.0 / 54.0;
    t[16] = 1.0 / 54.0;
    t[17] = 1.0 / 54.0;
    t[18] = 1.0 / 54.0;
    t[19] = 1.0 / 216.0;
    t[20] = 1.0 / 216.0;
    t[21] = 1.0 / 216.0;
    t[22] = 1.0 / 216.0;
    t[23] = 1.0 / 216.0;
    t[24] = 1.0 / 216.0;
    t[25] = 1.0 / 216.0;
    t[26] = 1.0 / 216.0;
    // bounce back directions
    bb[0] = 0;
    bb[1] = 2;
    bb[2] = 1;
    bb[3] = 4;
    bb[4] = 3;
    bb[5] = 6;
    bb[6] = 5;
    bb[7] = 8;
    bb[8] = 7;
    bb[9] = 10;
    bb[10] = 9;
    bb[11] = 12;
    bb[12] = 11;
    bb[13] = 14;
    bb[14] = 13;
    bb[15] = 16;
    bb[16] = 15;
    bb[17] = 18;
    bb[18] = 17;
    bb[19] = 20;
    bb[20] = 19;
    bb[21] = 22;
    bb[22] = 21;
    bb[23] = 24;
    bb[24] = 23;
    bb[25] = 26;
    bb[26] = 25;

    // discrete velocity
    e[0] = 0;
    e[1] = 0;
    e[2] = 0;

    e[3] = 1;
    e[4] = 0;
    e[5] = 0;

    e[6] = -1;
    e[7] = 0;
    e[8] = 0;

    e[9] = 0;
    e[10] = 1;
    e[11] = 0;

    e[12] = 0;
    e[13] = -1;
    e[14] = 0;

    e[15] = 0;
    e[16] = 0;
    e[17] = 1;

    e[18] = 0;
    e[19] = 0;
    e[20] = -1;

    e[21] = 1;
    e[22] = 1;
    e[23] = 0;

    e[24] = -1;
    e[25] = -1;
    e[26] = 0;

    e[27] = 1;
    e[28] = -1;
    e[29] = 0;

    e[30] = -1;
    e[31] = 1;
    e[32] = 0;

    e[33] = 1;
    e[34] = 0;
    e[35] = 1;

    e[36] = -1;
    e[37] = 0;
    e[38] = -1;

    e[39] = 1;
    e[40] = 0;
    e[41] = -1;

    e[42] = -1;
    e[43] = 0;
    e[44] = 1;

    e[45] = 0;
    e[46] = 1;
    e[47] = 1;

    e[48] = 0;
    e[49] = -1;
    e[50] = -1;

    e[51] = 0;
    e[52] = 1;
    e[53] = -1;

    e[54] = 0;
    e[55] = -1;
    e[56] = 1;

    e[57] = 1;
    e[58] = 1;
    e[59] = 1;

    e[60] = -1;
    e[61] = -1;
    e[62] = -1;

    e[63] = 1;
    e[64] = -1;
    e[65] = 1;

    e[66] = -1;
    e[67] = 1;
    e[68] = -1;

    e[69] = 1;
    e[70] = 1;
    e[71] = -1;

    e[72] = -1;
    e[73] = -1;
    e[74] = 1;

    e[75] = 1;
    e[76] = -1;
    e[77] = -1;

    e[78] = -1;
    e[79] = 1;
    e[80] = 1;

    buffer<int> ee{e};
    buffer<double> tt{t};

    auto ini1 = this->Q.submit([&](handler &h)
                               {
                                    size_t llx = lx;
                                    size_t lly = ly;
                                    size_t llz = lz;
                                    h.parallel_for(range{llx, lly, llz}, [=, fq = this->f, uq = this->ua, vq = this->va, wq = this->wa, pq = this->p, rhoq = this->rho](id<3> idx)
                                                   {
                                                    int id = idx[0] + idx[1] * llx + idx[2] * llx * lly;

                                                    uq[id] = 0.0;
                                                    vq[id] = 0.0;
                                                    wq[id] = 0.0;
                                                    pq[id] = 0.0;
                                                    rhoq[id] = 1.0;

                                              for (int ii = 0; ii < q;ii++){
                                                  fq[ii + id*q] = 0.0;

                          } }); });

    // generate random velocity
    for (int k = 0; k < lz; k++)
    {
        for (int j = 0; j < ly; j++)
        {
            for (int i = 0; i < lx; i++)
            {
                usr[i + j * lx + k * lx * ly] = (pow(x_lo + i - ghost - 32.0, 2) + pow(y_lo + j - ghost - 32.0, 2) + pow(z_lo + k - ghost - 32.0, 2) <= 32.0) ? 0 : 1;
                ua[i + j * lx + k * lx * ly] = (u0 * 4.0 * (z_lo + k - ghost) * (glz - 1 - (z_lo + k - ghost)) / (double)pow((glz - 1), 2)) * usr[i + j * lx + k * lx * ly] * (1.0 + (rand() % 100) * 0.001);
            }
        }
    }
    rdx = l_e[0] - ghost;
    rdy = l_e[1] - ghost;
    rdz = l_e[2] - ghost;

    auto ini2 = this->Q.submit([&](handler &h)
                               {
                                    accessor eee{ee, h, read_only};
                                    accessor ttt{tt, h, read_only};


                                    int llx = lx;
                                    int lly = ly;
                                    int llz = lz;
                                    h.parallel_for(range{rdx, rdy, rdz}, [=,fq=this->f,ftq=this->ft,uq=this->ua,vq=this->va,wq=this->wa,pq=this->p,tq=this->tau0](id<3> idx)
                                                {
                                             int id0 = idx[0] + ghost;
                                             int id1 = idx[1] + ghost;
                                             int id2 = idx[2] + ghost;
                                             int id = id0 + id1 * llx + id2 * llx * lly;

                                              for (int ii = 0; ii < q;ii++){

                                                  double eu2 = (eee[3 * ii] * uq[id] +
                                                                eee[3 * ii + 1] * vq[id] +
                                                                eee[3 * ii + 2] * wq[id]) *
                                                               (eee[3 * ii] * uq[id] +
                                                                eee[3 * ii + 1] * vq[id] +
                                                                eee[3 * ii + 2] * wq[id]);

                                                  double edu = eee[3 * ii] * uq[id] +
                                                               eee[3 * ii + 1] * vq[id] +
                                                               eee[3 * ii + 2] * wq[id];

                                                  double udu = uq[id] * uq[id] +
                                                               vq[id] * vq[id] +
                                                               wq[id] * wq[id];

                                                  double feq = ttt[ii] * pq[id] * 3.0 +
                                                               ttt[ii] * (3.0 * edu + 4.5 * eu2 - 1.5 * udu);

                                                  fq[ii + q * id] = ttt[ii] * pq[id] * 3.0 +
                                                                    ttt[ii] * (3.0 * edu + 4.5 * eu2 - 1.5 * udu);

                                                  ftq[ii + q * id] = 0.0;
                          } }); });

    MPI_Barrier(MPI_COMM_WORLD);
};

void LBM::Collision()
{
    buffer<int> ee{e};
    buffer<double> tt{t};
    rdx = l_e[0] - ghost;
    rdy = l_e[1] - ghost;
    rdz = l_e[2] - ghost;
    auto collision = this->Q.submit([&](handler &h)
                                    { 
                                    accessor eee{ee, h, read_only};
                                    accessor ttt{tt, h, read_only};
                                    int llx = lx;
                                    int lly = ly;
                                    int llz = lz;
                                    h.parallel_for(range{rdx, rdy, rdz}, [=,fq=this->f,uq=this->ua,vq=this->va,wq=this->wa,pq=this->p,tq=this->tau0](id<3> idx)
                                                     {
                                             int id0 = idx[0] + ghost;
                                             int id1 = idx[1] + ghost;
                                             int id2 = idx[2] + ghost;
                                             int id = id0 + id1 * llx + id2 * llx * lly;

                                             for (int ii = 0; ii < q;ii++){

                                                 double eu2 = (eee[3 * ii] * uq[id] +
                                                               eee[3 * ii + 1] * vq[id] +
                                                               eee[3 * ii + 2] * wq[id]) *
                                                              (eee[3 * ii] * uq[id] +
                                                               eee[3 * ii + 1] * vq[id] +
                                                               eee[3 * ii + 2] * wq[id]);

                                                 double edu = eee[3 * ii] * uq[id] +
                                                              eee[3 * ii + 1] * vq[id] +
                                                              eee[3 * ii + 2] * wq[id];

                                                 double udu = uq[id]*uq[id] +
                                                              vq[id]*vq[id] +
                                                              wq[id]*wq[id];

                                                 double feq = ttt[ii] * pq[id] * 3.0 +
                                                              ttt[ii] * (3.0 * edu + 4.5 * eu2 - 1.5 * udu);
                                                              
                                                fq[ii + q*id] -= (fq[ii+q*id] - feq) / (tq + 0.5);


                         } }); });
    this->Q.wait();
    MPI_Barrier(MPI_COMM_WORLD);
}

void LBM::Streaming()
{
    buffer<int> ee{e};
    buffer<double> tt{t};
    rdx = l_e[0] - ghost;
    rdy = l_e[1] - ghost;
    rdz = l_e[2] - ghost;

    if (y_lo == 0)
    {

        auto bcfr = this->Q.submit([&](handler &h)
                                   {
                                    accessor eee{ee, h, read_only};

                                    int llx = lx;
                                    int lly = ly;
                                    int llz = lz;
                                    h.parallel_for(range{ rdx+2, rdz+2}, [=,ls=this->l_s,fq=this->f,ftq=this->ft,bbq=this->bb](id<2> idx)
                                                     {
                                             int id0 = idx[0] + ghost-1;
                                             int id1 = ls[1]-1;
                                             int id2 = idx[1] + ghost-1;
                                             int id = id0 + id1 * llx + id2 * llx * lly;

                                             for (int ii = 0; ii < q;ii++){
                                                    if (eee[ii*3+1]>0)
                                                        {
                                                            int idt = (id0+2*eee[ii*3] ) + (id1+2 )*llx + (id2+ 2 * eee[ii * 3 + 2] )*llx*lly;
                                                            fq[ii+q*id] = fq[bbq[ii]+q*idt];
                                                        }

                         } }); });
    }
    this->Q.wait();
    if (y_hi == gly - 1)
    {

        auto bcba = this->Q.submit([&](handler &h)
                                   {
                                    accessor eee{ee, h, read_only};

                                    int llx = lx;
                                    int lly = ly;
                                    int llz = lz;
                                    h.parallel_for(range{ rdx+1, rdz+1}, [=,le=this->l_e,fq=this->f,ftq=this->ft,bbq=this->bb](id<2> idx)
                                                     {
                                             int id0 = idx[0] + ghost-1;
                                             int id1 = le[1];
                                             int id2 = idx[1] + ghost-1;
                                             int id = id0 + id1 * llx + id2 * llx * lly;

                                             for (int ii = 0; ii < q;ii++){
                                                    if (eee[ii*3+1]<0)
                                                        {
                                                            int idt = (id0+2*eee[ii*3] ) + (id1-2 )*llx + (id2+ 2 * eee[ii * 3 + 2] )*llx*lly;
                                                            fq[ii + q * id] = fq[bbq[ii]+ q*idt];
                                                        }

                         } }); });
    }

    this->Q.wait();
    if (z_lo == 0)
    {

        auto bcb = this->Q.submit([&](handler &h)
                                  {
                                    accessor eee{ee, h, read_only};

                                    int llx = lx;
                                    int lly = ly;
                                    int llz = lz;
                                    h.parallel_for(range{ rdx+1, rdy+1}, [=,ls=this->l_s,fq=this->f,ftq=this->ft,usrq=this->usr,bbq=this->bb](id<2> idx)
                                                     {
                                             int id0 = idx[0] + ghost-1;
                                             int id1 = idx[1] + ghost-1;
                                             int id2 = ls[2]-1;
                                             int id = id0 + id1 * llx + id2 * llx * lly;

                                             for (int ii = 0; ii < q;ii++){
                                                    if (eee[ii*3+2]>0)
                                                        {
                                                            int idt = (id0+2*eee[ii*3] ) + (id1+eee[ii*3+1] )*llx + (id2+2 )*llx*lly;
                                                            fq[ii + q * id] = fq[bbq[ii]+ q*idt];
                                                        }

                         } }); });
    }
    this->Q.wait();
    if (z_hi == glz - 1)
    {

        auto bct = this->Q.submit([&](handler &h)
                                  {
                                    accessor eee{ee, h, read_only};

                                    int llx = lx;
                                    int lly = ly;
                                    int llz = lz;
                                    h.parallel_for(range{ rdx+1, rdy+1}, [=,le=this->l_e,fq=this->f,ftq=this->ft,usrq=this->usr,bbq=this->bb](id<2> idx)
                                                     {
                                             int id0 = idx[0] + ghost-1;
                                             int id1 = idx[1] + ghost-1;
                                             int id2 = le[2];
                                             int id = id0 + id1 * llx + id2 * llx * lly;

                                             for (int ii = 0; ii < q;ii++){
                                                    if (eee[ii*3+2]<0)
                                                        {
                                                            int idt = (id0+2*eee[ii*3] ) + (id1+eee[ii*3+1] )*llx + (id2-2 )*llx*lly;
                                                            fq[ii + q * id] = fq[bbq[ii]+ q*idt];
                                                        }

                         } }); });
    }
    this->Q.wait();
    if (x_lo == 0)
    {

        auto bcl = this->Q.submit([&](handler &h)
                                  {
                                    accessor eee{ee, h, read_only};

                                    int llx = lx;
                                    int lly = ly;
                                    int llz = lz;
                                    h.parallel_for(range{ rdy+1, rdz+1}, [=,ls=this->l_s,fq=this->f,ftq=this->ft,usrq=this->usr,bbq=this->bb](id<2> idx)
                                                     {
                                             int id0 = ls[0]-1;
                                             int id1 = idx[0] + ghost-1;
                                             int id2 = idx[1] + ghost-1;
                                             int id = id0 + id1 * llx + id2 * llx * lly;

                                             for (int ii = 0; ii < q;ii++){
                                                    if (eee[ii*3]>0)
                                                        {
                                                            int idt = (id0+1 ) + (id1 )*llx + (id2 )*llx*lly;
                                                            fq[ii + q * id] = fq[ii+ q*idt];
                                                        }

                         } }); });
    }
    this->Q.wait();
    MPI_Barrier(MPI_COMM_WORLD);
    // right boundary free flow
    if (x_hi == glx - 1)
    {
        auto bcr = this->Q.submit([&](handler &h)
                                  {
                                    accessor eee{ee, h, read_only};

                                    int llx = lx;
                                    int lly = ly;
                                    int llz = lz;
                                    h.parallel_for(range{ rdy+1, rdz+1}, [=,le=this->l_e,fq=this->f,ftq=this->ft,usrq=this->usr,bbq=this->bb](id<2> idx)
                                                     {
                                             int id0 = le[0];
                                             int id1 = idx[0] + ghost-1;
                                             int id2 = idx[1] + ghost-1;
                                             int id = id0 + id1 * llx + id2 * llx * lly;

                                             for (int ii = 0; ii < q;ii++){
                                                    if (eee[ii*3]<0)
                                                        {
                                                            int idt = (id0-1 ) + (id1 + eee[ii * dim + 1])*llx + (id2 + eee[ii * dim + 2])*llx*lly;
                                                            fq[ii + q * id] = fq[ii+ q*idt];
                                                        }

                         } }); });
    }
    // streaming process
    this->Q.wait();
    MPI_Barrier(MPI_COMM_WORLD);

    auto usrb = this->Q.submit([&](handler &h)
                               {
                                    accessor eee{ee, h, read_only};

                                    int llx = lx;
                                    int lly = ly;
                                    int llz = lz;
                                    h.parallel_for(range{rdx, rdy, rdz}, [=,fq=this->f,ftq=this->ft,usrq=this->usr,bbq=this->bb](id<3> idx)
                                                     {
                                             int id0 = idx[0] + ghost;
                                             int id1 = idx[1] + ghost;
                                             int id2 = idx[2] + ghost;
                                             int id = id0 + id1 * llx + id2 * llx * lly;

                                             for (int ii = 0; ii < q;ii++){
                                                 int idt = (id0 + eee[ii * dim]) + (id1 + eee[ii * dim + 1])*llx + (id2 + eee[ii * dim + 2])*llx*lly;
                                                 int idt2 =  (id0 + 2*eee[ii * dim]) + (id1 + 2*eee[ii * dim + 1])*llx + (id2 + 2*eee[ii * dim + 2])*llx*lly;
                                                    if (usrq[id] == 0 && usrq[idt] == 1)
                                                        {
                                                            fq[ii + q * id] = fq[bbq[ii] + q*idt2];
                                                        }

                         } }); });
    this->Q.wait();
    auto stream1 = this->Q.submit([&](handler &h)
                                  {
                                    accessor eee{ee, h, read_only};

                                    int llx = lx;
                                    int lly = ly;
                                    int llz = lz;
                                    h.parallel_for(range{rdx, rdy, rdz}, [=,fq=this->f,ftq=this->ft](id<3> idx)
                                                     {
                                             int id0 = idx[0] + ghost;
                                             int id1 = idx[1] + ghost;
                                             int id2 = idx[2] + ghost;
                                             int id = id0 + id1 * llx + id2 * llx * lly;

                                             for (int ii = 0; ii < q;ii++){
                                                 int idt = (id0 - eee[ii * dim]) + (id1 - eee[ii * dim + 1])*llx + (id2 - eee[ii * dim + 2])*llx*lly;

                                                 ftq[ii + q * id] = fq[ii + q * idt];
                         } }); });
    this->Q.wait();
    auto stream2 = this->Q.submit([&](handler &h)
                                  {
                                    int llx = lx;
                                    int lly = ly;
                                    int llz = lz;
                                    h.parallel_for(range{rdx, rdy, rdz}, [=,fq=this->f,ftq=this->ft](id<3> idx)
                                                     {
                                             int id0 = idx[0] + ghost;
                                             int id1 = idx[1] + ghost;
                                             int id2 = idx[2] + ghost;
                                             int id = id0 + id1 * llx + id2 * llx * lly;
                                             for (int ii = 0; ii < q;ii++){

                                                 fq[ii + q * id] = ftq[ii + q * id];
                         } }); });
    this->Q.wait();
    MPI_Barrier(MPI_COMM_WORLD);
};

void LBM::Update()
{

    buffer<int> ee{e};
    buffer<double> tt{t};

    auto update = this->Q.submit([&](handler &h)
                                 { 
                                    accessor eee{ee, h, read_only};
                                    accessor ttt{tt, h, read_only};
                                    int llx = lx;
                                    int lly = ly;
                                    int llz = lz;
                                    int xl = x_lo;
                                    int zl = z_lo;
                                    int gz = glz;
                                    int xh = x_hi;
                                    int gx = glx;
                                    h.parallel_for(range{rdx, rdy, rdz}, [=, fq = this->f, uq = this->ua, vq = this->va, wq = this->wa, pq = this->p, usrq = this->usr,u0q=this->u0](id<3> idx)
                                                   {
                                                   int id0 = idx[0] + ghost;
                                                   int id1 = idx[1] + ghost;
                                                   int id2 = idx[2] + ghost;
                                                   int id = id0 + id1 * llx + id2 * llx * lly;

                                                   uq[id] = 0.0;
                                                   vq[id] = 0.0;
                                                   wq[id] = 0.0;
                                                   pq[id] = 0.0; 
                                                   for (int ii = 0; ii < q; ii++)
                                                    {

                                                        pq[id] = pq[id] + fq[ii + id * q ] / 3.0;
                                                        uq[id] = uq[id] + fq[ii + id * q] * eee[ii * 3];
                                                        vq[id] = vq[id] + fq[ii + id * q ] * eee[ii * 3 + 1];
                                                        wq[id] = wq[id] + fq[ii + id * q ] * eee[ii * 3 + 2];
                                                    }
                                                    uq[id] = uq[id] * usrq[id];
                                                    vq[id] = vq[id] * usrq[id];
                                                    wq[id] = wq[id] * usrq[id];
                                                    pq[id] = pq[id] * usrq[id]; 
                                                    
                                                    if (xl == 0)
                                                    {
                                                        uq[id-idx[0]] = u0q * 4.0 * (zl + idx[2]) * (gz - 1 - (zl + idx[2])) /(gz - 1)/(gz - 1);
                                                        vq[id-idx[0]] = 0.0;
                                                        wq[id-idx[0]] = 0.0;
                                                    }
                                                    if (xh == gx - 1)
                                                    {
                                                        pq[id1 * llx + id2 * llx * lly+llx-ghost-1] = 0.0;
                                                    } }); });

    this->Q.wait();

    MPI_Barrier(MPI_COMM_WORLD);
};

void LBM::MPIoutput(int n)
{
    // MPI_IO
    MPI_File fh;
    MPI_Status status;
    MPI_Offset offset = 0;

    MPI_Datatype FILETYPE, DATATYPE;
    // buffer
    int tp;
    float ttp;
    double fp;
    // min max
    double umin, umax, wmin, wmax, vmin, vmax, pmin, pmax;
    double uumin, uumax, wwmin, wwmax, vvmin, vvmax, ppmin, ppmax;

    // transfer
    double *uu, *vv, *ww, *pp, *xx, *yy, *zz;

    uu = (double *)malloc(ex * ey * ez * sizeof(double));
    vv = (double *)malloc(ex * ey * ez * sizeof(double));
    ww = (double *)malloc(ex * ey * ez * sizeof(double));
    pp = (double *)malloc(ex * ey * ez * sizeof(double));
    xx = (double *)malloc(ex * ey * ez * sizeof(double));
    yy = (double *)malloc(ex * ey * ez * sizeof(double));
    zz = (double *)malloc(ex * ey * ez * sizeof(double));

    for (int k = 0; k < ez; k++)
    {
        for (int j = 0; j < ey; j++)
        {
            for (int i = 0; i < ex; i++)
            {

                uu[i + j * ex + k * ey * ex] = ua[i + ghost + (j + ghost) * lx + (k + ghost) * lx * ly];
                vv[i + j * ex + k * ey * ex] = va[i + ghost + (j + ghost) * lx + (k + ghost) * lx * ly];
                ww[i + j * ex + k * ey * ex] = wa[i + ghost + (j + ghost) * lx + (k + ghost) * lx * ly];
                pp[i + j * ex + k * ey * ex] = p[i + ghost + (j + ghost) * lx + (k + ghost) * lx * ly];
                xx[i + j * ex + k * ey * ex] = (double)4.0 * (x_lo + i) / (glx - 1);
                yy[i + j * ex + k * ey * ex] = (double)(y_lo + j) / (gly - 1);
                zz[i + j * ex + k * ey * ex] = (double)(z_lo + k) / (glz - 1);
            }
        }
    }

    umin = *min_element(uu, uu + ex * ey * ez - 1);
    umax = *max_element(uu, uu + ex * ey * ez - 1);
    vmin = *min_element(vv, vv + ex * ey * ez - 1);
    vmax = *max_element(vv, vv + ex * ey * ez - 1);
    wmin = *min_element(ww, ww + ex * ey * ez - 1);
    wmax = *max_element(ww, ww + ex * ey * ez - 1);

    pmin = *min_element(pp, pp + ex * ey * ez - 1);
    pmax = *max_element(pp, pp + ex * ey * ez - 1);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&umin, &uumin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&umax, &uumax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&vmin, &vvmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&vmax, &vvmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&wmin, &wwmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&wmax, &wwmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&pmin, &ppmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&pmax, &ppmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    std::string str1 = "output" + std::to_string(n) + ".plt";
    const char *na = str1.c_str();
    std::string str2 = "#!TDV112";
    const char *version = str2.c_str();
    MPI_File_open(MPI_COMM_WORLD, na, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    if (comm.me == 0)
    {

        MPI_File_seek(fh, offset, MPI_SEEK_SET);
        // header !version number
        MPI_File_write(fh, version, 8, MPI_CHAR, &status);
        // INTEGER 1
        tp = 1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 3*4+8=20
        // variable name
        tp = 7;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 120;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 121;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 122;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 117;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 118;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 119;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 112;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 20+15*4=80
        // Zone Marker
        ttp = 299.0;
        MPI_File_write(fh, &ttp, 1, MPI_REAL, &status);
        // Zone Name
        tp = 90;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 79;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 78;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 69;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 32;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 48;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 48;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 49;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 80 + 10 * 4 = 120

        // Strand id
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // SOLUTION TIME
        double nn = (double)n;
        fp = nn;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ZONE COLOR
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ZONE TYPE
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // SPECIFY VAR LOCATION
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ARE RAW LOCAL
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // NUMBER OF MISCELLANEOUS
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ORDERED ZONE
        tp = glx;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = gly;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = glz;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // AUXILIARY
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // 120 + 13 * 4 = 172
        // EOHMARKER
        ttp = 357.0;
        MPI_File_write(fh, &ttp, 1, MPI_REAL, &status);
        // DATA SECTION
        ttp = 299.0;
        MPI_File_write(fh, &ttp, 1, MPI_REAL, &status);
        // VARIABLE DATA FORMAT
        tp = 2;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // PASSIVE VARIABLE
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // SHARING VARIABLE
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ZONE NUMBER
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // 172 + 12 * 4 = 220
        // MIN AND MAX VALUE FLOAT 64
        fp = 0.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 4.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 0.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 1.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 0.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 1.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = uumin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = uumax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = vvmin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = vvmax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = wwmin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = wwmax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = ppmin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = ppmax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);

        // 220 + 14 * 8 = 332
    }

    offset = 332;

    int glolen[3] = {glx, gly, glz};
    int localstart[3] = {x_lo, y_lo, z_lo};

    MPI_Type_create_subarray(dim, glolen, l_l, localstart, MPI_ORDER_FORTRAN, MPI_DOUBLE, &DATATYPE);

    // MPI_Type_commit(&DATATYPE);

    MPI_Type_contiguous(7, DATATYPE, &FILETYPE);

    MPI_Type_commit(&FILETYPE);

    MPI_File_set_view(fh, offset, MPI_DOUBLE, FILETYPE, "native", MPI_INFO_NULL);

    MPI_File_write_all(fh, xx, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, yy, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, zz, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, uu, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, vv, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, ww, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, pp, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);

    free(uu);
    free(vv);
    free(ww);
    free(pp);
    free(xx);
    free(yy);
    free(zz);

    MPI_Barrier(MPI_COMM_WORLD);
};
