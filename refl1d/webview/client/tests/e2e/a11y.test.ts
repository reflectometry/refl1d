import AxeBuilder from "@axe-core/playwright";
import { expect, panes, test } from "./fixtures";

/**
 * Known serious violations per panel, not yet fixed. Remove an entry once it's
 * fixed so the rule is enforced again. Moderate/minor rules (landmarks, an <h1>,
 * empty table headers) aren't enforced yet. The bumps panels are checked in bumps.
 */
const KNOWN: Record<string, Record<string, string>> = {
  Builder: {
    // TODO: every row reuses the same input ids, and the layer-name <label> is inside its <textarea>
    label: "layer name textarea has no label",
    "label-title-only": "layer name textarea is only labelled by its title",
  },
};

for (const title of ["Reflectivity", "Profile", "Builder"]) {
  test(`${title} panel has no serious a11y violations`, async ({ page }) => {
    const { left, right } = panes(page);
    // show the panel on the left, and a bumps panel that has passed bumps' own checks on the right
    await right.getByRole("tab", { name: "Log", exact: true }).click();
    const tab = left.getByRole("tab", { name: title, exact: true });
    await tab.click();
    await expect(tab).toHaveClass(/active/);
    await page.waitForTimeout(500); // let the panel fetch and draw

    const { violations } = await new AxeBuilder({ page }).analyze();
    const serious = violations
      .filter((v) => (v.impact === "serious" || v.impact === "critical") && !(v.id in (KNOWN[title] ?? {})))
      .map((v) => `${v.id} (${v.impact}): ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`);
    expect(serious).toEqual([]);
  });
}
