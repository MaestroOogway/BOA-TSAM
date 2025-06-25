% Use this editor as a MiniZinc scratch book
int: n;
int: m;
set of int: C = 1..n;
set of int: J = 1..m;

array[C,J] of int: dc;    % matriz de distancias cliente-hub
array[J] of int: f;       % costo fijo por hub
array[J] of int: L;       % capacidad por hub
int: Dmax;                % distancia máxima permitida

array[C,J] of var 0..1: x; % x[c,j] = 1 si cliente c va al hub j
array[J] of var 0..1: y;   % y[j] = 1 si se activa el hub j

var int: total_cost =
    sum(c in C, j in J)(dc[c,j] * x[c,j]) +  % distancia total
    sum(j in J)(f[j] * y[j]);                % costo fijo hubs

solve minimize total_cost;

constraint forall(c in C)( sum(j in J)(x[c,j]) = 1 );  % un cliente asignado a un solo hub
constraint forall(c in C, j in J)( x[c,j] <= y[j] );   % solo a hubs activos
constraint forall(j in J)( sum(c in C)(x[c,j]) <= L[j] ); % capacidad de cada hub
constraint forall(c in C, j in J where dc[c,j] > Dmax)( x[c,j] = 0 ); % respeta Dmax
