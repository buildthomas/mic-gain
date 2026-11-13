use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::{Duration, SystemTime};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{
    Device, FromSample, Host, SampleFormat, SizedSample, Stream, StreamConfig, SupportedStreamConfig,
};

const USAGE: &str = "\
mic-gain: boost a microphone and route it into a virtual audio device

USAGE:
    mic-gain                       Run using the settings file (created on first run)
    mic-gain --list                List audio devices
    mic-gain --compare <seconds>   Record, write mic-before.wav and mic-after.wav, print levels

OPTIONS (override the settings file for this run):
    --gain <dB>       Gain in decibels
    --input <name>    Part of the microphone's device name
    --output <name>   Part of the virtual device's name
    --config <path>   Use a different settings file

Settings file:
    Windows: %APPDATA%\\mic-gain\\mic-gain.conf
    macOS:   ~/Library/Application Support/mic-gain/mic-gain.conf
Edits to gain and compressor settings apply within a second while mic-gain is running.";

#[derive(Clone, PartialEq)]
struct Config {
    gain_db: f32,
    threshold_db: f32,
    ratio: f32,
    input: String,
    output: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            gain_db: 25.0,
            threshold_db: -12.0,
            ratio: 4.0,
            input: String::new(),
            output: String::new(),
        }
    }
}

#[derive(Default)]
struct Args {
    config: Option<PathBuf>,
    gain_db: Option<f32>,
    input: Option<String>,
    output: Option<String>,
    list: bool,
    compare: Option<f32>,
}

impl Args {
    fn apply(&self, mut c: Config) -> Config {
        if let Some(g) = self.gain_db {
            c.gain_db = g;
        }
        if let Some(i) = &self.input {
            c.input = i.clone();
        }
        if let Some(o) = &self.output {
            c.output = o.clone();
        }
        c
    }
}

struct Shared {
    buf: Mutex<VecDeque<f32>>,
    failed: AtomicBool,
    params: LiveParams,
}

/// Processing settings stored as f32 bits so they can change while audio runs.
struct LiveParams {
    gain: AtomicU32,
    threshold_db: AtomicU32,
    ratio: AtomicU32,
}

impl LiveParams {
    fn new(c: &Config) -> Self {
        let p = LiveParams {
            gain: AtomicU32::new(0),
            threshold_db: AtomicU32::new(0),
            ratio: AtomicU32::new(0),
        };
        p.set(c);
        p
    }

    fn set(&self, c: &Config) {
        self.gain.store(db_to_gain(c.gain_db).to_bits(), Ordering::Relaxed);
        self.threshold_db.store(c.threshold_db.to_bits(), Ordering::Relaxed);
        self.ratio.store(c.ratio.to_bits(), Ordering::Relaxed);
    }

    fn get(&self) -> (f32, f32, f32) {
        (
            f32::from_bits(self.gain.load(Ordering::Relaxed)),
            f32::from_bits(self.threshold_db.load(Ordering::Relaxed)),
            f32::from_bits(self.ratio.load(Ordering::Relaxed)),
        )
    }
}

struct Compressor {
    env: f32,
    attack: f32,
    release: f32,
}

impl Compressor {
    fn new(rate: u32) -> Self {
        let coef = |ms: f32| (-1.0 / (ms * 0.001 * rate as f32)).exp();
        Compressor {
            env: 0.0,
            attack: coef(5.0),
            release: coef(120.0),
        }
    }

    fn process(&mut self, x: f32, threshold_db: f32, ratio: f32) -> f32 {
        let level = x.abs();
        let c = if level > self.env { self.attack } else { self.release };
        self.env = c * self.env + (1.0 - c) * level;
        let over = 20.0 * self.env.max(1e-9).log10() - threshold_db;
        if over <= 0.0 || ratio <= 1.0 {
            return x;
        }
        x * db_to_gain(-over * (1.0 - 1.0 / ratio))
    }
}

fn describe_processing(c: &Config) -> String {
    let comp = if c.ratio <= 1.0 {
        "compressor off".to_string()
    } else {
        format!("compressor {}:1 above {} dBFS", c.ratio, c.threshold_db)
    };
    format!("{:+.1} dB gain, {comp}", c.gain_db)
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    let number = |v: Option<String>, flag: &str| -> Result<f32, String> {
        v.ok_or(format!("{flag} needs a number"))?
            .parse()
            .map_err(|_| format!("{flag} must be a number"))
    };
    while let Some(a) = it.next() {
        match a.as_str() {
            "--gain" => args.gain_db = Some(number(it.next(), "--gain")?),
            "--compare" => args.compare = Some(number(it.next(), "--compare")?),
            "--input" => args.input = Some(it.next().ok_or("--input needs a device name")?),
            "--output" => args.output = Some(it.next().ok_or("--output needs a device name")?),
            "--config" => args.config = Some(it.next().ok_or("--config needs a path")?.into()),
            "--list" => args.list = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0)
            }
            other => return Err(format!("unknown argument: {other}\n\n{USAGE}")),
        }
    }
    Ok(args)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2)
        }
    };
    let host = cpal::default_host();
    if args.list {
        list_devices(&host);
        return;
    }

    let path = args.config.clone().unwrap_or_else(default_config_path);
    let config = match load_config(&path) {
        Ok(c) => args.apply(c),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1)
        }
    };
    if let Some(secs) = args.compare {
        if let Err(e) = compare(&host, &config, secs) {
            eprintln!("{e}");
            std::process::exit(1)
        }
        return;
    }

    println!("Settings: {}", path.display());
    let mut last_reason = String::new();
    loop {
        let reason = match load_config(&path) {
            Ok(c) => run(&host, &args, &path, &args.apply(c)),
            Err(e) => e,
        };
        if reason != last_reason {
            eprintln!("{reason}; waiting...");
            last_reason = reason;
        }
        sleep(Duration::from_secs(2));
    }
}

fn default_config_path() -> PathBuf {
    let base = if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
    } else if cfg!(windows) {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
    };
    base.unwrap_or_else(|| PathBuf::from("."))
        .join("mic-gain")
        .join("mic-gain.conf")
}

fn config_template(c: &Config) -> String {
    format!(
        "# mic-gain settings. gain_db changes apply within a second while mic-gain runs;\n\
         # input/output changes make it reconnect. Run `mic-gain --list` for device names.\n\
         \n\
         # Boost in decibels\n\
         gain_db = {}\n\
         \n\
         # Compressor: above this level (dBFS), loud speech is turned down by the ratio.\n\
         # A ratio of 1 turns the compressor off.\n\
         compressor_threshold_db = {}\n\
         compressor_ratio = {}\n\
         \n\
         # Part of the microphone's device name; leave empty for the system's default mic\n\
         input = {}\n\
         \n\
         # Part of the virtual audio device's name; leave empty to use the first virtual cable found\n\
         output = {}\n",
        c.gain_db, c.threshold_db, c.ratio, c.input, c.output
    )
}

fn load_config(path: &Path) -> Result<Config, String> {
    if !path.exists() {
        let c = Config::default();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        }
        std::fs::write(path, config_template(&c))
            .map_err(|e| format!("write {}: {e}", path.display()))?;
        return Ok(c);
    }
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut c = Config::default();
    let mut has_gain = false;
    for (n, line) in text.trim_start_matches('\u{feff}').lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let at = || format!("{}:{}", path.display(), n + 1);
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("{}: expected key = value", at()))?;
        let value = value.trim().trim_matches('"').to_string();
        match key.trim() {
            "gain_db" => {
                c.gain_db = value
                    .parse()
                    .map_err(|_| format!("{}: gain_db must be a number", at()))?;
                has_gain = true;
            }
            "compressor_threshold_db" => {
                c.threshold_db = value
                    .parse()
                    .map_err(|_| format!("{}: compressor_threshold_db must be a number", at()))?
            }
            "compressor_ratio" => {
                c.ratio = value
                    .parse()
                    .ok()
                    .filter(|r: &f32| *r >= 1.0)
                    .ok_or_else(|| format!("{}: compressor_ratio must be a number of at least 1", at()))?
            }
            "input" => c.input = value,
            "output" => c.output = value,
            other => return Err(format!("{}: unknown setting '{other}'", at())),
        }
    }
    // Also guards against reading a half-saved file from an editor.
    if !has_gain {
        return Err(format!("{}: gain_db is missing", path.display()));
    }
    Ok(c)
}

fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

fn db_to_gain(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// Routes audio until something breaks or the devices change, then returns why.
fn run(host: &Host, args: &Args, path: &Path, config: &Config) -> String {
    let Some(input) = find_input(host, &config.input) else {
        return input_not_found(&config.input);
    };
    let Some(output) = find_output(host, &config.output) else {
        return if config.output.is_empty() {
            "no virtual audio device found".into()
        } else {
            format!("output device matching '{}' not found", config.output)
        };
    };
    let in_cfg = match input.default_input_config() {
        Ok(c) => c,
        Err(e) => return format!("input config: {e}"),
    };
    let out_cfg = match output.default_output_config() {
        Ok(c) => c,
        Err(e) => return format!("output config: {e}"),
    };

    let shared = Arc::new(Shared {
        buf: Mutex::new(VecDeque::new()),
        failed: AtomicBool::new(false),
        params: LiveParams::new(config),
    });
    let in_rate = in_cfg.sample_rate().0;
    let in_stream = match build_input(&input, &in_cfg, shared.clone()) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let out_stream = match build_output(&output, &out_cfg, shared.clone(), in_rate) {
        Ok(s) => s,
        Err(e) => return e,
    };
    if let Err(e) = in_stream.play() {
        return format!("start input: {e}");
    }
    if let Err(e) = out_stream.play() {
        return format!("start output: {e}");
    }

    println!(
        "Routing '{}' ({} Hz, {} ch) -> '{}' ({} Hz, {} ch) with {}",
        name(&input),
        in_rate,
        in_cfg.channels(),
        name(&output),
        out_cfg.sample_rate().0,
        out_cfg.channels(),
        describe_processing(config)
    );

    let (in_name, out_name) = (name(&input), name(&output));
    let mut current = config.clone();
    let mut stamp = modified(path);
    loop {
        sleep(Duration::from_secs(1));
        if shared.failed.load(Ordering::Relaxed) {
            return "audio stream stopped".into();
        }
        if !has_device(host.input_devices(), &in_name) {
            return "input device disconnected".into();
        }
        if !has_device(host.output_devices(), &out_name) {
            return "output device disconnected".into();
        }

        let now = modified(path);
        if now == stamp {
            continue;
        }
        stamp = now;
        match load_config(path) {
            Ok(c) => {
                let c = args.apply(c);
                if c.input != current.input || c.output != current.output {
                    return "device settings changed".into();
                }
                if c != current {
                    shared.params.set(&c);
                    println!("Now {}", describe_processing(&c));
                    current = c;
                }
            }
            Err(e) => eprintln!("{e} (keeping current settings)"),
        }
    }
}

fn name(d: &Device) -> String {
    d.name().unwrap_or_else(|_| "<unknown>".into())
}

fn find_device<I: Iterator<Item = Device>, E>(devices: Result<I, E>, pattern: &str) -> Option<Device> {
    let pattern = pattern.to_lowercase();
    devices
        .ok()?
        .find(|d| name(d).to_lowercase().contains(&pattern))
}

fn find_input(host: &Host, pattern: &str) -> Option<Device> {
    if !pattern.is_empty() {
        return find_device(host.input_devices(), pattern);
    }
    // Virtual cables often become the default input; recording from one would feed back into it.
    host.default_input_device()
        .filter(|d| !is_virtual(d))
        .or_else(|| host.input_devices().ok()?.find(|d| !is_virtual(d)))
}

fn find_output(host: &Host, pattern: &str) -> Option<Device> {
    if !pattern.is_empty() {
        return find_device(host.output_devices(), pattern);
    }
    host.output_devices().ok()?.find(is_virtual)
}

fn is_virtual(d: &Device) -> bool {
    let n = name(d).to_lowercase();
    n.contains("cable") || n.contains("blackhole")
}

fn input_not_found(pattern: &str) -> String {
    if pattern.is_empty() {
        "no microphone found".into()
    } else {
        format!("input device matching '{pattern}' not found")
    }
}

fn has_device<I: Iterator<Item = Device>, E>(devices: Result<I, E>, exact: &str) -> bool {
    devices.map(|mut d| d.any(|d| name(&d) == exact)).unwrap_or(false)
}

fn list_devices(host: &Host) {
    let describe = |c: Result<SupportedStreamConfig, cpal::DefaultStreamConfigError>| match c {
        Ok(c) => format!("{} Hz, {} ch, {:?}", c.sample_rate().0, c.channels(), c.sample_format()),
        Err(e) => e.to_string(),
    };
    println!("Input devices:");
    for d in host.input_devices().into_iter().flatten() {
        println!("  {}  [{}]", name(&d), describe(d.default_input_config()));
    }
    println!("Output devices:");
    for d in host.output_devices().into_iter().flatten() {
        println!("  {}  [{}]", name(&d), describe(d.default_output_config()));
    }
}

fn compare(host: &Host, config: &Config, secs: f32) -> Result<(), String> {
    let input = find_input(host, &config.input).ok_or_else(|| input_not_found(&config.input))?;
    let sup = input
        .default_input_config()
        .map_err(|e| format!("input config: {e}"))?;
    let rate = sup.sample_rate().0;
    let cfg = sup.config();
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let stream = match sup.sample_format() {
        SampleFormat::F32 => record_stream::<f32>(&input, &cfg, recorded.clone()),
        SampleFormat::I16 => record_stream::<i16>(&input, &cfg, recorded.clone()),
        SampleFormat::I32 => record_stream::<i32>(&input, &cfg, recorded.clone()),
        SampleFormat::U16 => record_stream::<u16>(&input, &cfg, recorded.clone()),
        f => Err(format!("unsupported input sample format {f:?}")),
    }?;
    stream.play().map_err(|e| format!("start input: {e}"))?;
    println!("Recording {secs} s from '{}'...", name(&input));
    sleep(Duration::from_secs_f32(secs));
    drop(stream);

    let gain = db_to_gain(config.gain_db);
    let mut comp = Compressor::new(rate);
    let mut limited = 0;
    let before = std::mem::take(&mut *recorded.lock().unwrap());
    let after: Vec<f32> = before
        .iter()
        .map(|&x| {
            let y = comp.process(x * gain, config.threshold_db, config.ratio);
            if y.abs() > 0.9 {
                limited += 1;
            }
            soft_clip(y)
        })
        .collect();
    write_wav("mic-before.wav", rate, &before)?;
    write_wav("mic-after.wav", rate, &after)?;

    println!("Settings: {}", describe_processing(config));
    print_levels("before", &before);
    print_levels("after", &after);
    println!(
        "{:.2}% of samples hit the limiter",
        100.0 * limited as f32 / before.len().max(1) as f32
    );
    println!("Wrote mic-before.wav and mic-after.wav");
    Ok(())
}

fn record_stream<T>(dev: &Device, cfg: &StreamConfig, out: Arc<Mutex<Vec<f32>>>) -> Result<Stream, String>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = cfg.channels as usize;
    dev.build_input_stream(
        cfg,
        move |data: &[T], _| {
            let mut out = out.lock().unwrap();
            for frame in data.chunks(channels) {
                let sum: f32 = frame.iter().map(|&s| s.to_sample::<f32>()).sum();
                out.push(sum / channels as f32);
            }
        },
        |e| eprintln!("input stream error: {e}"),
        None,
    )
    .map_err(|e| format!("open input: {e}"))
}

fn print_levels(label: &str, s: &[f32]) {
    let db = |x: f32| 20.0 * x.max(1e-9).log10();
    let peak = s.iter().fold(0f32, |m, x| m.max(x.abs()));
    let rms = (s.iter().map(|x| x * x).sum::<f32>() / s.len().max(1) as f32).sqrt();
    println!("{label:>16}: peak {:6.1} dBFS, RMS {:6.1} dBFS", db(peak), db(rms));
}

fn write_wav(path: &str, rate: u32, samples: &[f32]) -> Result<(), String> {
    let data_len = (samples.len() * 4) as u32;
    let mut b = Vec::with_capacity(44 + data_len as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * 4).to_le_bytes());
    b.extend_from_slice(&4u16.to_le_bytes());
    b.extend_from_slice(&32u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        b.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, b).map_err(|e| format!("write {path}: {e}"))
}

fn soft_clip(x: f32) -> f32 {
    const KNEE: f32 = 0.9;
    let a = x.abs();
    if a <= KNEE {
        x
    } else {
        x.signum() * (KNEE + (1.0 - KNEE) * ((a - KNEE) / (1.0 - KNEE)).tanh())
    }
}

fn build_input(dev: &Device, sup: &SupportedStreamConfig, shared: Arc<Shared>) -> Result<Stream, String> {
    let cfg = sup.config();
    match sup.sample_format() {
        SampleFormat::F32 => input_stream::<f32>(dev, &cfg, shared),
        SampleFormat::I16 => input_stream::<i16>(dev, &cfg, shared),
        SampleFormat::I32 => input_stream::<i32>(dev, &cfg, shared),
        SampleFormat::U16 => input_stream::<u16>(dev, &cfg, shared),
        f => Err(format!("unsupported input sample format {f:?}")),
    }
}

fn input_stream<T>(dev: &Device, cfg: &StreamConfig, shared: Arc<Shared>) -> Result<Stream, String>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = cfg.channels as usize;
    // Cap queued audio at 100 ms so latency can't grow if the output runs slow.
    let max_len = cfg.sample_rate.0 as usize / 10;
    let mut comp = Compressor::new(cfg.sample_rate.0);
    let err_shared = shared.clone();
    dev.build_input_stream(
        cfg,
        move |data: &[T], _| {
            let (gain, threshold_db, ratio) = shared.params.get();
            let mut buf = shared.buf.lock().unwrap();
            for frame in data.chunks(channels) {
                let sum: f32 = frame.iter().map(|&s| s.to_sample::<f32>()).sum();
                let y = comp.process(sum / channels as f32 * gain, threshold_db, ratio);
                buf.push_back(soft_clip(y));
            }
            let excess = buf.len().saturating_sub(max_len);
            buf.drain(..excess);
        },
        move |e| {
            eprintln!("input stream error: {e}");
            err_shared.failed.store(true, Ordering::Relaxed);
        },
        None,
    )
    .map_err(|e| format!("open input: {e}"))
}

fn build_output(
    dev: &Device,
    sup: &SupportedStreamConfig,
    shared: Arc<Shared>,
    in_rate: u32,
) -> Result<Stream, String> {
    let cfg = sup.config();
    match sup.sample_format() {
        SampleFormat::F32 => output_stream::<f32>(dev, &cfg, shared, in_rate),
        SampleFormat::I16 => output_stream::<i16>(dev, &cfg, shared, in_rate),
        SampleFormat::I32 => output_stream::<i32>(dev, &cfg, shared, in_rate),
        SampleFormat::U16 => output_stream::<u16>(dev, &cfg, shared, in_rate),
        f => Err(format!("unsupported output sample format {f:?}")),
    }
}

fn output_stream<T>(dev: &Device, cfg: &StreamConfig, shared: Arc<Shared>, in_rate: u32) -> Result<Stream, String>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = cfg.channels as usize;
    let step = in_rate as f64 / cfg.sample_rate.0 as f64;
    // Wait for 20 ms of audio before playing, so small timing jitter doesn't crackle.
    let prime_len = in_rate as usize / 50;
    let (mut a, mut b, mut frac, mut primed) = (0f32, 0f32, 0f64, false);
    let err_shared = shared.clone();
    dev.build_output_stream(
        cfg,
        move |data: &mut [T], _| {
            let mut buf = shared.buf.lock().unwrap();
            if !primed && buf.len() >= prime_len {
                primed = true;
            }
            for frame in data.chunks_mut(channels) {
                let v = if primed {
                    // Linear resampling in case the mic and virtual device run at different rates.
                    while frac >= 1.0 {
                        a = b;
                        b = buf.pop_front().unwrap_or_else(|| {
                            primed = false;
                            0.0
                        });
                        frac -= 1.0;
                    }
                    let v = a + (b - a) * frac as f32;
                    frac += step;
                    v
                } else {
                    0.0
                };
                let s = T::from_sample(v);
                frame.iter_mut().for_each(|o| *o = s);
            }
        },
        move |e| {
            eprintln!("output stream error: {e}");
            err_shared.failed.store(true, Ordering::Relaxed);
        },
        None,
    )
    .map_err(|e| format!("open output: {e}"))
}
