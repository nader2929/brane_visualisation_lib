# Visualisation
This library is to be used for visualising data.
## Functions

---

`pie_plot_column` : Create a plot and save it to a PNG file that is a pie chart displaying the categorical split of a column in the data provided. The image is saved to file at `/data/COLUMN_pie_plot.png`.

> Input

- `column : string` : The column to plot the sections of the pie chart based on its categorical values.

- `data_path : string` : The path of the data to use to plot.

> Output

- `resultOutput : string` : A string explaing the operation done and where the plot(s) have been saved.
---

`pie_plot_column_with_filter` : Create plots and save them to a PNG files that are pie charts displaying the categorical split of a column in the data provided and filtered by a second column. The image are saved to files at `/data/COLUMN_FILTER_COLUMN_x_distribution.png` and `/data/COLUMN_FILTER_COLUMN_rates.png`.

> Input

- `column : string` : The column to plot the sections of the pie chart based on its categorical values.

- `filter_column : string` : The column to filter main column on and seperate the categories on.

- `data_path : string` : The path of the data to use to plot.

- `legend : boolean` : Whether or not to include a legend in the plot. 


> Output

- `resultOutput : string` : A string explaing the operation done and where the plot(s) have been saved.
---

`scatter_plot_in_relation_to_column` : Create plots and save them to a PNG files that are scatter plots that have all the columns plotted on the X axis against a specfied column on the Y axis, while colouring the points based on a certain different column. The files are saved to files named `/data/COLUMN_XCOL_scatter.png`.

> Input

- `column : string` : The column to put on the Y axis and compare the distribution of all the other columns by plotting them on the X-axis.

- `data_path : string` : The path of the data to use to plot.

- `group_by_column : string` : The column by which to differentiate the colours of the points in the scatter plot.

- `alpha : real` : The transparency of the points in the scatter plot. 0 is completely transparent and 1 is completely solid.


> Output

- `resultOutput : string` : A string explaing the operation done and where the plot(s) have been saved.
---

`histogram_plot` : Create a plot and save it to a PNG file that is a histogram plot of a column in the provided dataset. It is also possible to split each value on the X-axs by another column. The resulting image will be saved to a `/data/COLUMN_histogram_plot.png`.

> Input

- `column : string` : The column to create the histogram of the categories or values of.

- `data_path : string` : The path of the data to use to plot.

- `hue_column : string` : The column by which to split the counts of, in order to compare the histogram distribution based on this column. To not split the histogram use the string " ".


> Output
- `resultOutput : string` : A string explaing the operation done and where the plot(s) have been saved.
---

`line_plot` : Create a plot and save it to a PNG file that is a line plot given a column to plot on the X-axis and a column to plot on the Y-axis, given a dataset. The resulting plot is saved to `/data/x_XCOLUMN_y_YCOLUMN_line_plot.png`.

> Input

- `x_column : string` : The column to use on the X-axis.

- `y_column : string` : The column to use on the Y-axis.

- `data_path : string` : The path of the data to use to plot. 


> Output
- `resultOutput : string` : A string explaing the operation done and where the plot(s) have been saved.
