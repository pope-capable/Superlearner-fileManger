'use strict';
import { DataTypes } from 'sequelize';
import db from '../../utils/db';
import {} from 'iyasunday';
import PredefinedFolder from '../predefinedFolder/model';

const TABLE = 'folders',
  schema = {
    userId: {
      type: DataTypes.INTEGER(),
      allowNull: false,
      validate: {
        notNull: { msg: 'Kindly suply userId' },
      },
    },

    name: {
      type: DataTypes.STRING(),
      allowNull: false,
    },

    slug: {
      type: DataTypes.STRING(),
      allowNull: false,
    },
  };

const Folder = db.define(TABLE, schema);
Folder.table = TABLE;

PredefinedFolder.hasMany(Folder, { onDelete: 'cascade' });
Folder.belongsTo(PredefinedFolder, { onDelete: 'cascade' });

export default Folder;
