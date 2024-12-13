use plotters::prelude::RGBColor;

/// Convert an HSV (0..360°, 0..1.0, 0..1.0) to RGB in 0..255 range
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let h = (h % 360.0 + 360.0) % 360.0; // normalize hue
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h / 60.0) as u8 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    let (r, g, b) = ((r + m) * 255.0, (g + m) * 255.0, (b + m) * 255.0);
    (r.round() as u8, g.round() as u8, b.round() as u8)
}

/// Iterator that yields distinct, non-primary colors.
struct DistinctColorIter {
    hue: f64,
    increment: f64,
}

impl DistinctColorIter {
    fn new() -> Self {
        // Start at some non-primary hue, e.g. 30 degrees
        // Use the golden angle (~137.5 degrees) to spread hues
        DistinctColorIter {
            hue: 30.0,
            increment: 137.50776405003785,
        }
    }
}

impl Iterator for DistinctColorIter {
    type Item = RGBColor;

    fn next(&mut self) -> Option<Self::Item> {
        // Avoid primary hues near 0°(red), 120°(green), 240°(blue)
        // We'll define a small function to shift hue if it's too close.
        fn adjust_hue(h: f64) -> f64 {
            let forbidden_hues = [0.0, 120.0, 240.0];
            let mut hue = h;
            for &fh in &forbidden_hues {
                let diff = ((hue - fh) + 360.0) % 360.0;
                // If hue is within 10 degrees of a forbidden hue, shift it by +15
                if diff < 10.0 || diff > 350.0 {
                    hue = (hue + 15.0) % 360.0;
                }
            }
            hue
        }

        self.hue = adjust_hue(self.hue);
        let (r, g, b) = hsv_to_rgb(self.hue, 1.0, 1.0);
        // Prepare next hue
        self.hue = (self.hue + self.increment) % 360.0;

        Some(RGBColor(r, g, b))
    }
}

/// Returns an iterator over distinct RGB colors avoiding pure red/green/blue.
fn distinct_colors() -> impl Iterator<Item = RGBColor> {
    DistinctColorIter::new()
}

fn main() {
    // Example usage:
    for c in distinct_colors().take(10) {
        println!("Color: {:?}", c);
    }
}
