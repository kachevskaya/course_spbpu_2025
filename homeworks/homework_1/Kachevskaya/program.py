import matplotlib
import math as mt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import odeint, solve_ivp
from json import loads, dumps



def calc_ws( gamma_wat: float) -> float:
    
    """
    Функция для расчета солесодержания в воде

    :param gamma_wat: относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.

    :return: солесодержание в воде, г/г
    """
    
    ws = (
            1 / (gamma_wat * 1000)
            * (1.36545 * gamma_wat * 1000 - (3838.77 * gamma_wat * 1000 - 2.009 * (gamma_wat * 1000) ** 2) ** 0.5)
    )
    # если значение отрицательное, значит скорее всего плотность ниже допустимой 992 кг/м3
    if ws > 0:
        return ws
    else:
        return 0


def calc_rho_w( ws: float, t: float) -> float:

    """
    Функция для расчета плотности воды в зависимости от температуры и солесодержания

    :param ws: солесодержание воды, г/г
    :param t: температура, К

    :return: плотность воды, кг/м3
    """
    
    rho_w = (1000 * (1.0009 - 0.7114 * ws + 0.2605 * ws ** 2) ** (-1))/(1 + (t - 273) * 1e-4 * (0.269 * (t - 273) ** 0.637 - 0.8))

    return rho_w 

def calc_mu_w( ws: float, t: float, p: float) -> float:
    
    a = (
            109.574
            - (0.840564 * 1000 * ws)
            + (3.13314 * 1000 * ws ** 2)
            + (8.72213 * 1000 * ws ** 3)
    )
    b = (
            1.12166
            - 2.63951 * ws
            + 6.79461 * ws ** 2
            + 54.7119 * ws ** 3
            - 155.586 * ws ** 4
    )

    mu_w = (
            a * (1.8 * t - 460) ** (-b)
            * (0.9994 + 0.0058 * (p * 1e-6) + 0.6534 * 1e-4 * (p * 1e-6) ** 2)
    )
    return mu_w


def calc_n_re( rho_w: float, q_ms: float, mu_w: float, d_tub: float) -> float:
    v = q_ms / (np.pi * d_tub ** 2 / 4)
    n_re = rho_w * v * d_tub / mu_w * 1000
    
    return n_re


def calc_ff_churchill(n_re: float, roughness: float, d_tub: float) -> float:
        
    a = (-2.457 * np.log((7 / n_re) ** 0.9 + 0.27 * (roughness / d_tub))) ** 16
    b = (37530 / n_re) ** 16

    ff = 8 * ((8 / n_re) ** 12 + 1 / (a + b) ** 1.5) ** (1/12)
    return ff


def calc_ff_churchill(n_re: float, roughness: float, d_tub: float) -> float:

    a = (-2.457 * np.log((7 / n_re) ** 0.9 + 0.27 * (roughness / d_tub))) ** 16
    b = (37530 / n_re) ** 16

    ff = 8 * ((8 / n_re) ** 12 + 1 / (a + b) ** 1.5) ** (1/12)
    return ff

def calc_ff_jain(n_re: float, roughness: float, d_tub: float) -> float:

    if n_re < 3000:
        ff = 64 / n_re
    else:
        ff = 1 / (1.14 - 2 * np.log10(roughness / d_tub + 21.25 / (n_re**0.9))) ** 2
    return ff

def calc_dp_dl_grav(rho_w: float, angle: float):
    dp_dl_grav = rho_w * 9.81 * np.sin(angle / 180 * np.pi)
    return dp_dl_grav


def calc_dp_dl_fric( rho_w: float, mu_w: float, q_ms: float, d_tub: float, roughness: float):
    
    if q_ms != 0:
        n_re = calc_n_re(rho_w, q_ms, mu_w, d_tub)
        ff = calc_ff_churchill(n_re, roughness, d_tub)
        dp_dl_fric = ff * rho_w * q_ms ** 2 / d_tub ** 5
    else:
        dp_dl_fric = 0
    return dp_dl_fric

def calc_dp_dl( rho_w: float, mu_w: float, angle: float, q_ms: float, d_tub: float, roughness: float) -> float:

    dp_dl_grav = calc_dp_dl_grav(rho_w, angle)

    dp_dl_fric = calc_dp_dl_fric(rho_w, mu_w, q_ms, d_tub, roughness)

    dp_dl = dp_dl_grav - 0.815 * dp_dl_fric

    return dp_dl


import json

data = {"gamma_water": 0.9974119149798719, "md_vdp": 2764.0549899214448, "d_tub": 0.06591752424146266, "angle": 80.47848386669185, "roughness": 0.0003091727046321148, "p_wh": 136.05798700778203, "t_wh": 26.104332293188772, "temp_grad": 2.2827422270429416}

def integr_func( h: float, pt: tuple, temp_grad: float, gamma_wat: float, angle: float, q_ms: float, d_tub: float, roughness: float ) -> tuple:
   
    return calc_dp_dl(calc_rho_w(calc_ws(gamma_wat), pt[1]), calc_mu_w(calc_ws(gamma_wat), pt[1], pt[0]), angle, q_ms, d_tub, roughness), temp_grad


def calc_pipe( p_wh: float, t_wh: float, h0: float, md_vdp: float, temp_grad: float, gamma_wat: float, angle: float, q_ms: float, d_tub: float, roughness: float ) -> tuple:

    num_of_steps = 1000 
    step_size = (md_vdp - h0) / num_of_steps

    p = [p_wh]
    t = [t_wh]

    depths = np.linspace(h0, md_vdp, num_of_steps)
    
    #Метод Эйлера-Коши
    for i in range(len(depths) - 1):
        step_size = depths[i + 1] - depths[i]
        
        #dp_i/dx_i
        grad_p, grad_t = integr_func(depths[i], (p[-1], t[-1]), temp_grad, gamma_wat, angle, q_ms, d_tub, roughness)
        #dp_i+1 с волной / dx_i+1
        grad_pp, grad_tt = integr_func(depths[i + 1], (p[-1] + grad_p * step_size, t[-1] + grad_t * step_size), temp_grad, gamma_wat, angle, q_ms, d_tub, roughness)
        
        p.append(p[-1] + (grad_p + grad_pp) / 2 * step_size)
        t.append(t[-1] + (grad_t + grad_tt) / 2 * step_size)
    
    return p, t


def calc_p_wf(p_wh: float, t_wh: float, h0: float, md_vdp: float, temp_grad: float, gamma_wat: float, angle: float, q_ms: float, d_tub: float, roughness: float ) -> float:

    p = calc_pipe(p_wh, t_wh, h0, md_vdp, temp_grad, gamma_wat, angle, q_ms, d_tub, roughness)[0]

    return p[-1]


Q = []
for t in range(1, 400, 25):
    Q.append(t)

gamma_water = data['gamma_water']
H = data['md_vdp']
d_tub = data['d_tub']
angle = data['angle']
roughness = data['roughness']
p_wh = data['p_wh'] * 101325 
t_wh = data['t_wh'] + 273 
temp_grad = data['temp_grad']

P = []

for q in Q:
        q_ms = q / 86400 
        p_wf = calc_p_wf(p_wh, t_wh, 0, H, temp_grad, gamma_water, angle, q_ms, d_tub, roughness) / 101325
        P.append(p_wf)
        data_export = {
            "q_liq": Q,
            "p_wf": P
        }

with open('output.json', 'w') as json_file:
    json.dump(data_export, json_file, indent=4)


plt.plot(Q, P)
plt.title("Зависимость давления от дебита")
plt.ylabel("Давление")
plt.xlabel("Дебит")



