fs = 200;

kalman_gain = 0.3


% Bode of: y[t] = (1-K) y[t-1] + K x[t]
% H(z) = K / (1 - (1-K) z^-1)

K  = 0.3;                 % <-- set your K in (0,1]
Ts = 1/fs;                    % <-- sample time (sec). set to your dt.

a = 1 - K;

% Discrete-time transfer function (z^-1 form)
Hz = tf([K], [1 -a], Ts);  % numerator: K, denominator: 1 - a z^-1

fc = K/(2*pi*Ts);


% Choose a frequency grid (rad/s) up to (slightly below) Nyquist
wN = pi/Ts;
w  = logspace(log10(1e-0), log10(0.99*wN), 500);  % rad/s

[m,p,wout] = bode(Hz, w);      % m: abs, p: deg, wout: rad/s
m = squeeze(m);
p = squeeze(p);
f = wout/(2*pi);               % Hz

% Plot manually
figure;
subplot(211);
semilogx(f, 20*log10(m)); grid on; hold on;
xline(fc,'r--','LineWidth',1.5);
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');

subplot(212);
semilogx(f, p); grid on; hold on;
xline(fc,'r--','LineWidth',1.5);
xlabel('Frequency (Hz)'); ylabel('Phase (deg)');

sgtitle(['MN Filter, fc = ' num2str(fc), ' Hz']);

fc

K  = 0.001;                 % <-- set your K in (0,1]
Ts = 1/fs;                    % <-- sample time (sec). set to your dt.

a = 1 - K;

% Discrete-time transfer function (z^-1 form)
Hz = tf([K], [1 -a], Ts);  % numerator: K, denominator: 1 - a z^-1

fc = K/(2*pi*Ts);


% Choose a frequency grid (rad/s) up to (slightly below) Nyquist
wN = pi/Ts;
w  = logspace(log10(1e-3), log10(0.99*wN), 500);  % rad/s

[m,p,wout] = bode(Hz, w);      % m: abs, p: deg, wout: rad/s
m = squeeze(m);
p = squeeze(p);
f = wout/(2*pi);               % Hz

% Plot manually
figure;
subplot(211);
semilogx(f, 20*log10(m)); grid on; hold on;
xline(fc,'r--','LineWidth',1.5);
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');

subplot(212);
semilogx(f, p); grid on; hold on;
xline(fc,'r--','LineWidth',1.5);
xlabel('Frequency (Hz)'); ylabel('Phase (deg)');

sgtitle(['Hebbian Filter, fc = ' num2str(fc), ' Hz']);


fc
