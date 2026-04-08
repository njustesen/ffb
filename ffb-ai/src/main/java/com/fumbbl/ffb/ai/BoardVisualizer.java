package com.fumbbl.ffb.ai;

import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.ai.client.AiClient;
import com.fumbbl.ffb.client.FieldComponent;
import com.fumbbl.ffb.client.PitchDimensionProvider;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Map;

/**
 * Saves a PNG showing the current field state with a probability overlay.
 *
 * The base image is taken from the rendered {@link FieldComponent} (all layers
 * composited). For each candidate square, a semi-transparent colour coded by
 * probability is drawn on top, along with the probability percentage.
 *
 * Files are saved to /tmp/ffb-board-{timestamp}.png.
 */
public final class BoardVisualizer {

    private BoardVisualizer() {}

    /**
     * Save a board snapshot.
     * Colors and labels both come from {@code probabilities}, normalized by max for color.
     */
    public static void save(AiClient client, String label, Map<FieldCoordinate, Double> probabilities) {
        save(client, label, probabilities, null);
    }

    /**
     * Save a board snapshot with a highlighted player square.
     * Colors and labels both come from {@code probabilities}, normalized by max for color.
     *
     * @param highlightCoord coordinate of the acting player to highlight (may be null)
     */
    public static void save(AiClient client, String label, Map<FieldCoordinate, Double> probabilities,
                            FieldCoordinate highlightCoord) {
        save(client, label, probabilities, probabilities, highlightCoord);
    }

    /**
     * Save a board snapshot with separate color and label maps.
     *
     * @param colorValues    map used for the red-green gradient (normalized by max internally)
     * @param labelValues    map used for the text label in each square (shown as %; may be same as colorValues)
     * @param highlightCoord coordinate of the acting player to highlight (may be null)
     */
    public static void save(AiClient client, String label,
                            Map<FieldCoordinate, Double> colorValues,
                            Map<FieldCoordinate, Double> labelValues,
                            FieldCoordinate highlightCoord) {
        if (client == null || colorValues == null || colorValues.isEmpty()) {
            return;
        }
        try {
            FieldComponent fc = client.getUserInterface().getFieldComponent();
            fc.refresh();
            BufferedImage base = fc.getImage();
            if (base == null) {
                return;
            }

            BufferedImage out = new BufferedImage(base.getWidth(), base.getHeight(),
                BufferedImage.TYPE_INT_ARGB);
            Graphics2D g = out.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.drawImage(base, 0, 0, null);

            PitchDimensionProvider pdp = client.getUserInterface().getPitchDimensionProvider();
            int sq = pdp.fieldSquareSize();

            // Normalize colour map by max so the best square is always full green.
            double maxColor = colorValues.values().stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
            double colorScale = (maxColor > 0.0) ? 1.0 / maxColor : 1.0;

            for (Map.Entry<FieldCoordinate, Double> entry : colorValues.entrySet()) {
                FieldCoordinate coord = entry.getKey();
                float norm = (float) (entry.getValue() * colorScale);
                Dimension screenPos = pdp.mapToLocal(coord, false);
                int px = screenPos.width;
                int py = screenPos.height;

                // Interpolate red → green by normalized colour value
                Color fill = interpolate(
                    new Color(200, 0, 0, 150),
                    new Color(0, 200, 0, 180),
                    norm);
                g.setColor(fill);
                g.fillRect(px, py, sq, sq);

                // White border
                g.setColor(new Color(255, 255, 255, 100));
                g.drawRect(px, py, sq - 1, sq - 1);

                // Label: use labelValues for text (shown as %)
                Double labelVal = labelValues != null ? labelValues.get(coord) : null;
                if (labelVal != null) {
                    double pct = labelVal * 100.0;
                    String lbl = pct < 0.05 ? "<0.1%"
                        : pct < 1.0 ? String.format("%.1f%%", pct)
                        : String.format("%.0f%%", pct);
                    g.setColor(Color.WHITE);
                    g.setFont(new Font("SansSerif", Font.BOLD, Math.max(8, sq / 4)));
                    g.drawString(lbl, px + 2, py + sq - 4);
                }
            }

            // Highlight the acting player square with a cyan border
            if (highlightCoord != null) {
                Dimension hPos = pdp.mapToLocal(highlightCoord, false);
                int hx = hPos.width;
                int hy = hPos.height;
                g.setColor(new Color(0, 220, 255, 230));
                int border = Math.max(2, sq / 10);
                for (int b = 0; b < border; b++) {
                    g.drawRect(hx + b, hy + b, sq - 1 - 2 * b, sq - 1 - 2 * b);
                }
            }

            // Decision label at top-left
            g.setColor(Color.YELLOW);
            g.setFont(new Font("SansSerif", Font.BOLD, 14));
            g.drawString(label != null ? label : "", 10, 20);
            g.dispose();

            String path = "/tmp/ffb-board-" + System.currentTimeMillis() + ".png";
            ImageIO.write(out, "png", new File(path));
        } catch (IOException ex) {
            System.err.println("[BoardVisualizer] Failed to save: " + ex.getMessage());
        } catch (Exception ex) {
            System.err.println("[BoardVisualizer] Unexpected error: " + ex.getMessage());
        }
    }

    private static Color interpolate(Color from, Color to, float t) {
        float clamped = Math.max(0f, Math.min(1f, t));
        int r = (int) (from.getRed()   + clamped * (to.getRed()   - from.getRed()));
        int g = (int) (from.getGreen() + clamped * (to.getGreen() - from.getGreen()));
        int b = (int) (from.getBlue()  + clamped * (to.getBlue()  - from.getBlue()));
        int a = (int) (from.getAlpha() + clamped * (to.getAlpha() - from.getAlpha()));
        return new Color(r, g, b, a);
    }
}
