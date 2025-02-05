# AI_Decoded
 XAI group assignment


The **COMPAS dataset** (Correctional Offender Management Profiling for Alternative Sanctions) is used to assess the risk of recidivism (reoffending) among individuals in the criminal justice system. Below is an explanation of each column:

### **Identifiers & Demographics:**
- **`id`** – A unique identifier for each individual.
- **`name`** – Full name of the individual.
- **`first`** – First name.
- **`last`** – Last name.
- **`sex`** – Gender of the individual (e.g., Male, Female).
- **`dob`** – Date of birth.
- **`age`** – Age of the individual at the time of assessment.
- **`age_cat`** – Categorical age grouping (e.g., "<25", "25-45", "45+").
- **`race`** – The race of the individual (e.g., African-American, Caucasian, Hispanic).

### **Juvenile & Criminal History:**
- **`juv_fel_count`** – Number of juvenile felony offenses.
- **`juv_misd_count`** – Number of juvenile misdemeanor offenses.
- **`juv_other_count`** – Number of other juvenile offenses.
- **`priors_count`** – Total number of prior offenses.

### **Case Information & Risk Assessment:**
- **`days_b_screening_arrest`** – Number of days between the arrest and the COMPAS screening.
- **`c_jail_in`** – Date the individual was jailed for the current offense.
- **`c_jail_out`** – Release date from jail for the current offense.
- **`c_days_from_compas`** – Days between the offense and COMPAS screening.
- **`c_charge_degree`** – Severity of the current charge (M = Misdemeanor, F = Felony).
- **`c_charge_desc`** – Description of the current charge.
- **`is_recid`** – Whether the individual reoffended (1 = Yes, 0 = No).
- **`r_charge_degree`** – Degree of the re-offense charge (if applicable).
- **`r_days_from_arrest`** – Days between the first arrest and the re-offense.
- **`r_offense_date`** – Date of the re-offense.
- **`r_charge_desc`** – Description of the re-offense.
- **`r_jail_in`** – Jail entry date for the re-offense.

### **Violent Recidivism:**
- **`violent_recid`** – Indicates whether the individual committed a violent re-offense.
- **`is_violent_recid`** – Whether the re-offense was violent (1 = Yes, 0 = No).
- **`vr_charge_degree`** – Degree of the violent re-offense charge.
- **`vr_offense_date`** – Date of the violent re-offense.
- **`vr_charge_desc`** – Description of the violent re-offense.

### **COMPAS Risk Assessment Scores:**
- **`type_of_assessment`** – Type of COMPAS assessment (e.g., "Risk of Recidivism").
- **`decile_score`** – COMPAS recidivism risk score (1-10).
- **`decile_score.1`** – Another instance of the risk score (likely redundant or related to different assessments).
- **`score_text`** – Risk category based on decile score (e.g., "Low", "Medium", "High").
- **`screening_date`** – Date when the COMPAS assessment was conducted.
- **`v_type_of_assessment`** – Type of assessment specifically for violent recidivism.
- **`v_decile_score`** – COMPAS violent recidivism risk score (1-10).
- **`v_score_text`** – Risk category for violent recidivism.

### **Additional/Repetitive Columns:**
- **`priors_count.1`** – Another instance of `priors_count` (likely redundant).
- **`event`** – Possibly a binary variable indicating whether recidivism occurred.
