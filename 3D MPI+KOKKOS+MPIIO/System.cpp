#include "System.hpp"

void System::Initialize()
{
    // system defination
    this->cs2 = 1.0 / 3.0;
    this->cs = sqrt(cs2);

    this->rho0 = 1.0;
    this->R = 0.1 * this->sy;
    this->Re = 100;
    this->u0 = 0.1;
    this->Ma = this->u0 / this->cs;
    this->miu = this->rho0 * this->u0 * this->R / this->Re;
    this->Time = 10000;
    this->inter = 1000;
    this->tau = this->u0 * this->R / Re / cs2;
};

void System::Monitor()
{

    std::cout << "3D Cylinder Flow" << std::endl
              << "Re    =" << this->Re << std::endl
              << "Ma    =" << this->Ma << std::endl
              << "rho   =" << this->rho0 << std::endl
              << "miu   =" << this->miu << std::endl
              << "tau   =" << this->tau << std::endl
              << "Time  =" << this->Time << std::endl
              << "inter =" << this->inter << std::endl
              << "============================" << std::endl;
};
