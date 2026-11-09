import { expect, panes, test } from "./fixtures";

// The server is started with doc/examples/ex1/nifilm-fit.py: a Ni film on Si, measured at four angles.
const ANGLES = 4;

test("Tabs include the Refl1D panels in place of the bumps ones", async ({ page }) => {
  const tabs = panes(page).left.getByRole("tab");
  await expect(tabs.first()).toHaveText("Reflectivity");
  const titles = await tabs.allTextContents();
  expect(titles.map((t) => t.trim())).toEqual(expect.arrayContaining(["Reflectivity", "Profile", "Builder"]));
  expect(titles.map((t) => t.trim())).not.toContain("Data");
});

test("Reflectivity panel draws theory and data for every angle", async ({ page }) => {
  const { left } = panes(page);
  await expect(left.getByRole("tab", { name: "Reflectivity", exact: true })).toHaveClass(/active/);
  // a line (theory) and markers (data) per angle
  await expect(left.locator(".scatterlayer .trace")).toHaveCount(2 * ANGLES);
  await expect(left.locator(".annotation-text")).toContainText("2 =");
});

for (const mode of ["Fresnel (R/R_substrate)", "RQ^4", "Reflectivity"]) {
  test(`Reflectivity panel draws in ${mode} mode`, async ({ page }) => {
    const { left } = panes(page);
    await left.locator("#plot_mode").selectOption(mode);
    await expect(left.locator(".g-ytitle")).toContainText(mode === "RQ^4" ? "Q" : mode.split(" ")[0]!);
    await expect(left.locator(".scatterlayer .trace")).toHaveCount(2 * ANGLES);
  });
}

test("Profile panel draws the SLD profile", async ({ page }) => {
  const { right } = panes(page);
  await right.getByRole("tab", { name: "Profile", exact: true }).click();
  await expect(right.locator("#model option")).not.toHaveCount(0);
  await expect(right.locator(".scatterlayer .trace").first()).toBeVisible();
});

test("Builder creates an editable model from an unsupported loaded model", async ({ page }) => {
  const { right } = panes(page);
  await right.getByRole("tab", { name: "Builder", exact: true }).click();
  await expect(right.getByText("Unsupported model")).toBeVisible();
  page.once("dialog", (dialog) => dialog.accept());
  await right.getByRole("button", { name: "New Model" }).click();

  const names = right.locator("tbody tr textarea");
  await expect(names).toHaveCount(2);
  await expect(names.nth(0)).toHaveValue("Si");
  await expect(names.nth(1)).toHaveValue("Vacuum");
});

for (const title of ["Reflectivity", "Summary", "Profile", "Log", "History", "Convergence", "Model", "Parameters"]) {
  test(`${title} panel opens without errors`, async ({ page }) => {
    const { left } = panes(page);
    const tab = left.getByRole("tab", { name: title, exact: true });
    await tab.click();
    await expect(tab).toHaveClass(/active/);
    await expect(left.locator(".tab-content > *").first()).toBeVisible();
    // give async fetch-and-draw a moment to fail, if it's going to
    await page.waitForTimeout(500);
  });
}
