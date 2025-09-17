I created this project to try and predict the fantasy performance of NFL quarterbacks points per game using both statistical modeling and real world context.

I collected historical QB statistical data from fantasy pros such as yards, touchdowns, interceptions, etc and cleaned the data using Google Sheets and converted it into a CSV file.

Using this data, I used linear regression to establish a baseline prediction of how quarterbacks may perform simply based on the assumption of repeating a similar season.

However, I knew that it is fairly unrealistic to assume a quarterback will produce the same way, so I built an adjustment model to factor external changes.

These changes included things like team change, coaching change, offensive line strength, schedule difficulty, amount of indoor vs outdoor games, returning from an injury, and even off field considerations such as social factors.

By combining my baseline prediction with these adjustments, the model produces projections that can predict the fanatasy point production of any quarterback.

Now while this model currently works for established quarterbacks, there are still areas I am looking to improve.

One of the biggests challenges is predicting performance of rookies or quarterbacks who just got the starting role.

To do this I plan to build a system that uses college stats and preseason stats.

I also plan to adjust the model even more by incorporating more complex machine learning algorithms.
