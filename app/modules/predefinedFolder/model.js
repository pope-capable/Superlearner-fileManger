import { DataTypes } from 'sequelize';
import db from '../../utils/db';

const schema = {
  id: {
    type: DataTypes.TINYINT().UNSIGNED,
    primaryKey: true,
    autoIncrement: true,
    allowNull: false,
  },

  name: {
    type: DataTypes.CHAR(),
    unique: true,
    allowNull: false,
    validate: {
      notNull: { msg: 'Kindly enter foler name' },
    },
  },
};
const table = 'predefined_folders';

const PredefinedFolder = db.define(table, schema);
PredefinedFolder.table = table;

export default PredefinedFolder;
